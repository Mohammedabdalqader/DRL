import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from collections import deque,namedtuple

import torch.optim as optim
from ddpg_model import Actor, Critic



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    '''Interact with and learn from environment.'''

    def __init__(self, state_size, action_size,seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.t_step = 0 # counter for activating learning every few steps
        self.TAU    = 1e-2
        self.gamma  = 0.99
        self.BUFFER_SIZE = int(1e6)
        self.BATCH_SIZE  = 1024
        self.LR_CRITIC   = 1e-3
        self.LR_ACTOR = 1e-3
        self.WEIGHT_DECAY = 0.0
        self.EPSILON = 1.0
        self.EPSILON_DECAY = 0.99
        
        # Actor network (w/ target network)
        self.actor_local = Actor(self.state_size, self.action_size, seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR_ACTOR)

        # Critic network (w/ target network)
        self.critic_local = Critic(self.state_size, self.action_size, seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.LR_CRITIC, weight_decay=self.WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(self.action_size, self.seed)


    def act(self, state,add_noise = True):
        """ Given a state choose an action
        Params
        ======
            state (float ndarray): state of the environment        
        """
        
        state = torch.from_numpy(state).unsqueeze(0).float().to(device)
        self.actor_local.eval() # set network on eval mode, this has any effect only on certain modules (Dropout, BatchNorm, etc.)
        with torch.no_grad():
            action = self.actor_local(state).cpu().squeeze(0).data.numpy()
                        
        self.actor_local.train() # set nework on train mode
        if add_noise:
            action += self.noise.sample()*self.EPSILON
                
        return np.clip(action, -1, 1)
        

    def reset(self):
   
        self.noise.reset()

    def learn(self, experiences):
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
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)              
            
        self.EPSILON *= self.EPSILON_DECAY
        self.noise.reset()
        
        
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