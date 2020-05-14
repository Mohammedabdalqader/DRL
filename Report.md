[//]: # (Image References)

[dqn]: images/dqn.jpg "dqn"
[model]: images/DQNvsDueling.png "model"

# Navigation

In this report I will explain everything about this project in details. So we will look at different aspects like:
- **Deep Q-Larning**
- **Deep Q-Learning improvements**
- **Model architectures**
- **Hayperparameters**
- **Further work**


### Deep Q-Learning

In deep Q learning we use a neural network to approximate the Q-value function. The network receives the state as input  and outputs the Q-values for all possible actions. The largest output is our next action. 

![dqn][dqn]

Before we train our DQN, we have to deal with an issue that plays a crucial role in how the agent learns to estimate Q-values, and that is `Experience Replay`

`Experience replay` is a concept where we help the agent to remember and not forget its previous actions by replaying them. Every once in a while, we sample a batch of previous experiences (which are stored in a buffer) and we feed the network. That way the agent relives its past and improve its memory. Experience Replay is based on the idea that we can learn better, if we do multiple passes over the same experience and to generate uncorelatted experience data for online training of deep RL agents.

So far so good. But of course, there are a few problems that arise such as :

**Moving Q-Targets**

the first component of the TD Error is the Q-Target and it is calculated as the immediate reward plus the discounted max Q-value for the next state. When we train our agent, we update the weights accordingly to the TD Error. But the same weights apply to both the target and the predicted value. 

We move the output closer to the target, but we also move the target. So, we end up chasing the target and we get a highly oscillated training process. 

`Solution`  :  Fixed Q-Targets

Instead of using one Neural Network, it uses two. 
One as the main Deep Q Network and a second one Target Network. The weights of the target updated only once in a while.

**Overestimation of Q-values**

Since Q values are very noisy, when you take the max over all actions, you're probably getting an overestimated value.
Think that if for some reason the network overestimates a Q value for an action, that action will be chosen as the go-to action for the next step and the same overestimated value will be used as a target value. In other words, there is no way to evaluate if the action with the max value is actually the best action.

`Solution`  :  Double Deep Q Network

The Double Q-learning algorithm is an adaption of the DQN algorithm that reduces the observed overestimation, and also leads to much better performance on several Atari games.

The idea of Double Q-learning is to reduce overestimations by decomposing the max operation in the target into action selection and action evaluation.

ِAlso to address Overestimation of Q-values, we use two Deep Q Networks:
- One of these DQNs is responsible for the selection of the next action (the one with the maximum value) as always.
- The second one (Target network) is responsible for the evaluation of that action.


Now I will mention other techniques that can be used to improve the original DQN :smiley:	

### Deep Q-Learning improvements

There are different extensions to the Deep Q-Networks (DQN) algorithm such as : 
- `Double DQN` 
- `Dueling DQN`
- `Prioritized Experience Replay`
- `multi-step bootstrap targets`
- `Distributional DQN`
- `Noisy DQN`

Each of the six extensions address a different issue with the original DQN algorithm.

What if we combine all these six modifications and use them to train RL agents, will we get good results? :thinking:	
The answer is : Yes 

Researchers at Google DeepMind recently tested the performance of an agent that incorporated all six of these modifications. The corresponding algorithm was termed `Rainbow`.

It outperforms each of the individual modifications and achieves state-of-the-art performance on Atari 2600 games!


### Model architectures

![model][model]

