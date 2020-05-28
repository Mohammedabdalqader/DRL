[//]: # (Image References)

[actor-critic]: Collaboration_and_Competition/images/actor-critic.png "AC"
[maddpg]: Collaboration_and_Competition/images/maddpg.png "MADDPG"
[maddpg_arch]: Collaboration_and_Competition/images/maddpg_arch.png "MADDPG_ARCH"


# Multi-Agent Collaboration and Competition

In this report I will explain everything about this project in details. So we will look at different aspects like:

- **Actor-Critic**
- **MADDPG (Multi-Agent Deep Deterministic Policy Gradient) algorithm**
- **Model architectures**
- **Hayperparameters**
- **Training Process**
- **Result**
- **Future Work**


### Actor-Critic

Actor-critical algorithms are the basis behind almost every modern RL method like PPO, A3C and many more. So to understand all these new techniques, you definitely need a good understanding of what actor-critic is and how it works.

Let us first distinguish between value-based and policy-based methods:

Value-based methods like Q-Learning and its extensions try to find or approximate the optimal value function, which is a mapping between an action and a value, while policy-based methods like Policy Gradients and REINFORCE try to find the optimal policy directly without the Q value.

Each method has its advantages. For example, policy-based methods are better suited for continuous and stochastic environments, have faster convergence, while value-based methods are more efficient and stable.

Actor-critics aim to take advantage of all the good points of both the value-based and the policy-based while eliminating all their disadvantages.  

The basic idea is to split the model into two parts: one to calculate an action based on a state and another to generate the Q-values of the action. 

![AC][actor-critic]


Actor  : decides which action to take

Critic : tells the actor how good its action was and how it should adjust.

### MADDPG (Multi-Agent Deep Deterministic Policy Gradient) algorithm

To achieve the desired average score, a multi-agent DDPG (deep deterministic Policy Gradient) actor-critic architecture was chosen.

![MADDPG_ARCH][maddpg_arch]

Similar to the "Actor Critic" architecture with only one agent, each agent has its own network of actors and critics. The input for the actor network is the current state of the agent and the output is an suitable action for that agent in that state. The critic part, however, is slightly different from the usual single agent DDPG. Here the critic network of each agent has full visibility on the environment. It records not only the observation and action of this particular agent, but also the observations and actions of all other agents. (Collaboration and Competition Situation)

But what I have implemented For this project is a **competitive version** of MADDPG (Multi-Agent Deep Deterministic Policy Gradient). Each agent has its own DDPG actor-critic architecture and does not communicate with other agents (but sharing the same Memory) and the goal of each agent is to maximize their own returns.

The reason why I have chosen this algorithm is that this algorithm has achieved a good result on various problems

### Model architectures

Regarding the model I have experimented too much with it, I have tried different architecture (one of them was taken from this [paper](https://arxiv.org/pdf/1509.02971.pdf))and here are some of them:

|  | # Hidden Layers(Actor)|# neurons (Actor)| # Hidden Layers(Critic) |# neurons (Critic)| Batch Normalization|
| ---------- | ---------- |---------- |---------- |---------- |---------- |
|1|2|[400,300]|2|[400,300]|:x:|
|2|2|[400,300]|2|[400,300]|:heavy_check_mark:|
|3|2|[400,300]|3|[400,300,300]|:heavy_check_mark:|
|3|2|[400,300]|4|[128,64,64,32]|:heavy_check_mark:|
|4|3|[400,300,300]|4|[128,64,64,32]|:heavy_check_mark: Best Results|


**Actor Architecture**

Both Actor-Networks (local and target) for each agent consist of 4 fully-connected layers ( 3 hidden layers, 1 output layer) each of hidden layers followed by Batch Normalization layer and Relu activation function.

The number of neurons of the fully-connected layers are as follows:

- fc1 , number of neurons: 400,
- fc2 , number of neurons: 300,
- fc3 , number of neurons: 300,
- fc4 , number of neurons: 2 (number of actions)


**Critic Architecture**

Both Critic-Networks (local and target) for each agent consist of 5 fully-connected layers ( 4 hidden layers, 1 output layer) each of hidden layers followed by a Relu activation function.

The number of neurons of the fully-connected layers are as follows:

- fc1 , number of neurons: 128,
- fc2 , number of neurons: 64,
- fc3 , number of neurons: 64,
- fc4 , number of neurons: 32,
- output , number of neurons: 1,


### Hyperparameters

There were many hyperparameters involved in the experiment. The value of each of them is given below:

| Hyperparameter                      | Value |
| ----------------------------------- | ----- |
| Replay buffer size                  | 1e6  |
| Batch size                          | 256  |
| discount factor          | 0.99  |
| TAU                              | 1e-2  |
| Actor Learning rate                 | 1e-3  |
| Critic Learning rate                | 1e-3  |
| Update interval                     | 1    |
| LEARN_NUMBER        | 4    |
| Number of episodes                  | 250 (max)   |
| epsilon start | 1.0 |
| epsilon decay | 0.99 |

# Training Process

I will now explain the training process in detail.

1- I have defined a class called maddpg that takes (state_size,action_size,num_agents) as input and prepares the models for each agent and some necessary functions.


```
class maddpg():
    
    def __init__(self,state_size,action_size,num_agents,random_seeds):
        
        self.state_size   = state_size
        self.action_size  = action_size
        self.num_agents   = num_agents
        self.random_seeds = random_seeds
        self.agents = [Agent(self.state_size,self.action_size,random_seeds[i]) for i in range(self.num_agents)]
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed = 7)
        
    def act(self,states,add_noise = True):
        
        actions = [agent.act(state,add_noise) for agent,state in zip(self.agents,states)]
        return actions

    def reset(self):
        for i in range(self.num_agents):
            self.agents[i].reset()
            
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        #for idx, agent in enumerate(self.maddpg_agent):
            self.memory.add(state, action, reward, next_state, done)


        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            for agent in self.agents:
                for _ in range(LEARN_NUMBER):
                    experiences = self.memory.sample()
                    agent.learn(experiences)
```

2- In class **Agent()** i initialize actor-critic model for each agents

```
class Agent():
    '''Interact with and learn from environment.'''

    def __init__(self, state_size, action_size,seed):
        .
        .
        .
        
        # Actor network (w/ target network)
        self.actor_local = Actor(self.state_size, self.action_size, seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR_ACTOR)

        # Critic network (w/ target network)
        self.critic_local = Critic(self.state_size, self.action_size, seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.LR_CRITIC, weight_decay=self.WEIGHT_DECAY)
        .
        .
        .
```

3- Actor-Critic Model

```
class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_units=[400,300,300]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        

        # Hidden Layers 
        self.fc1 = nn.Linear(state_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc4 = nn.Linear(hidden_units[2], action_size)
        self.reset_parameters()
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(hidden_units[0])
        self.bn2 = nn.BatchNorm1d(hidden_units[1])
        self.bn3 = nn.BatchNorm1d(hidden_units[2])

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))

        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.bn1(self.fc1(states)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return F.tanh(self.fc4(x))
        
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_units=[128,64,64,32]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        
        # Hidden Layers
        self.fcs1 = nn.Linear(state_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0] + action_size , hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc4 = nn.Linear(hidden_units[2], hidden_units[3])
        self.fc5 = nn.Linear(hidden_units[3], 1)
        self.reset_parameters()

        # Batch Normalization
        self.bn = nn.BatchNorm1d(hidden_units[0])


    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)


    def forward(self, states, actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x_ = F.relu(self.bn(self.fcs1(states)))
        x  = torch.cat((x_, actions), dim=1)
        x  = F.relu(self.fc2(x))
        x  = F.relu(self.fc3(x))
        x  = F.relu(self.fc4(x))

        return self.fc5(x)

```

4- Befor training the agents i intialize the agents models and all other function.

```
m_agent = maddpg(state_size=state_size, action_size=action_size,num_agents=num_agents, random_seeds=[0,1])

```

5- Now it's training time.

  * at the beginning of each episode I reset the environment.
  * each agent will receive his own local observation and act with the environment and then estimate the best action (Local Actor Network) 
  * After each agent has estimated the action to be taken... some noise is added to this action and then each agent will take this action and receive (new_state,reward,done(wither the epsiode has finished or not and some info)
  * Each agent adds its experience to the replay buffer (memory sharing)
  * Is it time to update the models? Take some experience from memory sharing and train the local actor critical model for each agent, then update the traget actor critical model with soft update.
  * Iterate until we reach the desired average reward which is in our case  +0.5 over 100 consecutive episodes, after taking the maximum over both agents.
  


# Results
The desired average reward is achieved after 196 episodes.

| MADDPG (Multi-Agent Deep Deterministic Policy Gradient)|
| ---------- |
|![MADDPG][maddpg]|

# Future Work

After 2 months with the excellent knowledge that this course has given us, I can say that I have taken a big step towards mastering this area. I am able to implement different deep reinforcement learning algorithms and to select a suitable one for each problem.
In this project i have a chieved a very good result, in less than 200 episodes the target average reward achieved (> 0.50) and in 250 episode the average reward was 1.338 :muscle:. but I wonder if the performance will be better if I use **prioritized experience replay**? So I will work on it, and if it gives a better result, I will share the results with you :grinning:

In this environment, the agents can also work together to keep the ball in play as long as possible and get more rewards. Therefore I will implement the idea of the **mixed cooperative competing multi-agent DDPG** as mentioned in the paper [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)
