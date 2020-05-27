[//]: # (Image References)

[actor-critic]: Continuous-Control/images/actor-critic.png "ac"


# Continuous Control

While working on this project, I had to invest too much time in research to find the right algorithms for such a problem. There were many options available to me, and this was a challenge for me, and from here my journey began.

There is really a very useful [repo](https://github.com/ShangtongZhang/DeepRL) that describes and implements different algorithms that work very well for such a problem with continuous action space. Thanks to this repo and other sources, I was able to understand some algorithms correctly, including the DDPG, D4PG, PPO, A2C, and A3C algorithms, and I was able to implement some of these algorithms to solve my problem.

In this report I will explain everything about this project in details. So we will look at different aspects like:
- **Actor-Critic**
- **Distributed distributional deep deterministic policy gradients (D4PG) algorithm**
- **Model architectures**
- **Hayperparameters**
- **Future Work**


### Actor-Critic

Actor-critical algorithms are the basis behind almost every modern RL method like PPO, A3C and so on. So to understand all these new techniques, you definitely need a good understanding of what actor-critic is and how it works.

Let us first distinguish between value-based and policy-based methods:

Value-based methods like Q-Learning and its extensions try to find or approximate the optimal value function, which is a mapping between an action and a value, while policy-based methods like Policy Gradients and REINFORCE try to find the optimal policy directly without the Q value.

Each method has its advantages. For example, policy-based methods are better suited for continuous and stochastic environments, have faster convergence, while value-based methods are more efficient and stable.

Actor-critics aim to take advantage of all the good points of both the value-based and the policy-based while eliminating all their disadvantages.  

![ac][actor-critic]
The basic idea is to split the model into two parts: one to calculate an action based on a state and another to generate the Q-values of the action. 


