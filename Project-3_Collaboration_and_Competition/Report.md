[//]: # (Image References)

[actor-critic]: Collaboration_and_Competition/images/actor-critic.png "AC"
[maddpg]: Collaboration_and_Competition/images/maddpg.png "MADDPG"
[maddpg_arch]: Collaboration_and_Competition/images/maddpg_arch.png "MADDPG_ARCH"


# Multi-Agent Collaboration and Competition

For this project I have trained my Agents with MADDPG (Multi-Agent Deep Deterministic Policy Gradient) algorithm in a competitive environment where the goal of each agent is to maximize its own returns.

In this report I will explain everything about this project in details. So we will look at different aspects like:

- **Actor-Critic**
- **MADDPG (Multi-Agent Deep Deterministic Policy Gradient) algorithm**
- **Model architectures**
- **Hayperparameters**
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



But what I have implemented here is a competitive version of multi-agent DDPG where the goal of each agent is to maximize their own returns.


### Model architectures

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

# Results
| MADDPG (Multi-Agent Deep Deterministic Policy Gradient)|
| ---------- |
|![MADDPG][maddpg]|

# Future Work

After 2 months with the excellent knowledge that this course has given us, I can say that I have taken a big step towards mastering this area. I am able to implement different algorithms and to select a suitable one for each problem.
In this project i have a chieved a very good result, in less than 200 episodes the target average reward achieved (> 0.50) and in 250 episode the average reward was 1.338 :muscle:. but I wonder if the performance will be better if I use prioritized experience replay? So I will work on it, and if it gives a better result, I will share the results with you :grinning:
