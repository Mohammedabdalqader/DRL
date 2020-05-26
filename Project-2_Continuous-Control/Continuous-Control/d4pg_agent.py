import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from collections import deque,namedtuple

from l2_projection import _l2_project
import torch.optim as optim
from d4pg_model import Actor, Critic
from prioritized_memory import PrioritizedMemory

LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
WEIGHT_DECAY = 0.0     # L2 weight decay
BATCH_SIZE = 256
BUFFER_SIZE = int(1e5)
UPDATE_EVERY = 1
UPDATE_FACTOR = 1
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    '''Interact with and learn from environment.'''

    def __init__(self, state_size, action_size,num_agents,seed):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents    = num_agents
        self.seed = random.seed(seed)
        self.v_min = -10.0
        self.v_max = 10.0
        self.num_atoms = 51
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.t_step = 0 # counter for activating learning every few steps


        # Actor network (w/ target network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic network (w/ target network)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.critic_criterion = nn.BCELoss()
        # Noise process: note that each agent has an independent process
        self.noise = OUNoise(action_size, self.seed)
        
        # Replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, self.seed)

        # Prioritized replay memory
        # self.prioritized_memory = PrioritizedMemory(BATCH_SIZE, BUFFER_SIZE, self.seed)

    def act(self, state, add_noise = True):
        """ Given a state choose an action
        Params
        ======
            state (float ndarray): state of the environment        
        """
        
        state = torch.from_numpy(state).float().to(device)
        
        self.actor_local.eval() # set network on eval mode, this has any effect only on certain modules (Dropout, BatchNorm, etc.)
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
                        
        self.actor_local.train() # set nework on train mode
        if add_noise:
            action += self.noise.sample()
                
        return np.clip(action, -1, 1)


    def step(self, states, actions, rewards, next_states, dones):
        # add new experience in memory
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

            
        '''   
        if len(self.prioritized_memory) >= BATCH_SIZE:
            min_learning = len(self.prioritized_memory) // BATCH_SIZE
            num_learning = np.min([self.num_agents, min_learning])
            
            for i in range(num_learning):
                update_target_net = False
                idxes, experiences, is_weights = self.prioritized_memory.sample(device)
                if (i + 1) % UPDATE_EVERY == 0:
                    update_target_net = True
                    self.learn(experiences, GAMMA, is_weights=is_weights, leaf_idxes=idxes,update_target_net=update_target_net)
        '''
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                n_train = np.min([UPDATE_FACTOR,len(self.memory)//BATCH_SIZE])
                for _ in range(n_train):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)


    def reset(self):

        self.noise.reset()

    def learn(self, experiences, gamma):
        """
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Policy loss = (1/n)*Q_local(s,a) -> for deterministic policy (no log prob)
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            is_weights (tensor array): importance-sampling weights for prioritized experience replay
            leaf_idxes (numpy array): indexes for update priorities in SumTree
        """

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #

        crt_distr_v = self.critic_local(states, actions)
        last_act_v = self.actor_target(next_states)

        last_distr_v = F.softmax(self.critic_target(next_states, last_act_v), dim=1)
        
        proj_distr_v = _l2_project(next_distr_v=last_distr_v,
                                         rewards_v=rewards,
                                         dones_mask_t=dones,
                                         gamma=gamma ** 5,
                                         n_atoms=self.num_atoms,
                                         v_min=self.v_min,
                                         v_max=self.v_max,
                                         delta_z=self.delta_z)
        
        proj_distr_v = torch.FloatTensor(proj_distr_v).to(device)
        prob_dist_v = -F.log_softmax(crt_distr_v, dim=1) * proj_distr_v

        critic_loss_v = prob_dist_v.sum(dim=1).mean()
        

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss_v.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) # clip gradient to max 1
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        cur_actions_v = self.actor_local(states)
        crt_distr_v = self.critic_local(states, cur_actions_v)
        actor_loss_v = -self.critic_local.distr_to_q(crt_distr_v)
        actor_loss_v = actor_loss_v.mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss_v.backward()
        self.actor_optimizer.step()

        # ------------------- update target network ------------------- #
   
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)