[//]: # (Image References)

[actor-critic]: ../Continuous-Control/images/actor-critic.png "AC"
[maddpg]: Collaboration_and_Competition/images/maddpg.png "MADDPG"


# Multi-Agent Collaboration and Competition

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

![ac][actor-critic]


Actor  : decides which action to take

Critic : tells the actor how good its action was and how it should adjust.

### Distributed distributional deep deterministic policy gradients (D4PG) algorithm

The core idea in this algorithm is to replace a single Q-value from the critic with N_ATOMS values, corresponding to the probabilities of values from the pre-defined range. The Bellman equation is replaced with the Bellman operator, which transforms this distributional representation in a similar way.

### Model architectures

**Actor Architecture**

Both Actor-Networks (local and target) consist of 3 fully-connected layers ( 2 hidden layers, 1 output layers) each of hidden layers followed by a Relu activation function and Batch Normalization layer.

The number of neurons of the fully-connected layers are as follows:

- fc1 , number of neurons: 400,
- fc2 , number of neurons: 300,
- fc3 , number of neurons: 4 (number of actions),

**Critic Architecture**

Both Critic-Networks (local and target) consist of 3 fully-connected layers ( 2 hidden layers, 1 output layers) each of hidden layers followed by a Relu activation function.

The number of neurons of the fully-connected layers are as follows:

- fc1 , number of neurons: 400,
- fc2 , number of neurons: 300,
- fc3 , number of neurons: 51 (number of atoms),


### Hyperparameters

There were many hyperparameters involved in the experiment. The value of each of them is given below:

| Hyperparameter                      | Value |
| ----------------------------------- | ----- |
| Replay buffer size                  | 1e5   |
| Batch size                          | 256  |
| discount factor          | 0.99  |
| TAU                              | 1e-3  |
| Actor Learning rate                 | 1e-3  |
| Critic Learning rate                | 1e-3  |
| Update interval                     | 1    |
| Update times per interval           | 1    |
| Number of episodes                  | 2000 (max)   |
| Max number of timesteps per episode | 1000  |
| Number of atoms                  | 51  |
| Vmin | -10  |
| Vmax | +10  |


# Results
| MADDPG (Multi-Agent Deep Deterministic Policy Gradient)|
| ---------- |
|![MADDPG][result]|

# Future Work

After 2 months with the excellent knowledge that this course has given us, I can say that I have taken a big step towards mastering this area. I am able to implement different algorithms and to select a suitable one for each problem.
In this project i have a chieved a very good result, in less than 200 episodes the target average reward achieved (> 0.50) and in 250 episode the average reward was 1.338 :muscle:. but I wonder if the performance will be better if I use prioritized experience replay? So I will work on it, and if it gives a better result, I will share the results with you :grinning:
