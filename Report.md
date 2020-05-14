[//]: # (Image References)

[dqn]: dqn.jpg "dqn"

# Navigation

In this report I will explain everything about this project in details. So we will look at different aspects like:
- **Deep Q-Larning**
- **Deep Q-Learning improvements**
- **Model architectures**
- **Hayperparameters**
- **further work**


### Deep Q-Learning

In deep Q learning we use a neural network to approximate the Q-value function. The network receives the state as input  and outputs the Q-values for all possible actions. The largest output is our next action. 

![dqn][dqn]

Before we train out DQN, we have to deal with an issue that plays a crucial role in how the agent learns to estimate Q-values, and that is `Experience Replay`

`Experience replay` is a concept where we help the agent to remember and not forget its previous actions by replaying them. Every once in a while, we sample a batch of previous experiences (which are stored in a buffer) and we feed the network. That way the agent relives its past and improve its memory. Experience Replay is based on the idea that we can learn better, if we do multiple passes over the same experience and to generate uncorelatted experience data for online training of deep RL agents.

So far so good. But of course, there are a few problems that arise such as :

**Moving Q-Targets**

the first component of the TD Error is the Q-Target and it is calculated as the immediate reward plus the discounted max Q-value for the next state. When we train our agent, we update the weights accordingly to the TD Error. But the same weights apply to both the target and the predicted value. 

We move the output closer to the target, but we also move the target. So, we end up chasing the target and we get a highly oscillated training process. 

`Solution`  
Instead of using one Neural Network, it uses two. 
One as the main Deep Q Network and a second one Target Network. The weights of the target updated only once in a while. This technique is called Fixed Q-Targets.

**Overestimation of Q-values**

