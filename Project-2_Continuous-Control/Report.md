[//]: # (Image References)

[actor-critic]: Continuous-Control/images/actor-critic.png "ac"
[d4pg]: Continuous-Control/images/d4pg.png "d4pg"


# Continuous Control

In this report I will explain everything about this project in details. So we will look at different aspects like:
- **Actor-Critic**
- **Distributed distributional deep deterministic policy gradients (D4PG) algorithm**
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


### Result

![d4pg][d4pg]


### Future Work

While working on this project, I had to invest too much time in research to find the right algorithms for such a problem. There were many options available to me, and this was a challenge for me, and from here my journey began.

There is really a very useful [repo](https://github.com/ShangtongZhang/DeepRL) that describes and implements different algorithms that work very well for such a problem with continuous action space. Thanks to this repo and other sources, I was able to understand some algorithms correctly, including the DDPG, D4PG, PPO, A2C, and A3C algorithms, and I was able to implement some of these algorithms to solve my problem.

Here are some Ideas for improvement:

* Implementing TRPO, PPO, A3C, A2C algorithms:

  It is worthwhile to implement all these algorithms, so I will work on it in the next days and see which of these algorithms converges   faster. 

* Adjusting the Hyperparameters:

  The more important step I can also take to improve the results and solve the problem with 100 episodes or even < 100 is to adjust the   hyper parameters. 

* Using prioritized experience replay and N-step techniques:

  As mentioned in this paper https://openreview.net/forum?id=SyZipzbCb using techniques with D4PG could potentially lead to better  results

