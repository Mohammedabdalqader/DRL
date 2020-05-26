[//]: # (Image References)

[dqn]: Navigation/images/dqn.jpg "dqn"
[model]: Navigation/images/DQNvsDueling.png "model"
[result]: Navigation/results/DQN.png "DQN"
[result1]: Navigation/results/Double.png "Double"
[result2]: Navigation/results/Dueling.png "Dueling"
[result3]: Navigation/results/PER.png "PER"

# Navigation

In this report I will explain everything about this project in details. So we will look at different aspects like:
- **Deep Q-Larning**
- **Deep Q-Learning improvements**
- **Model architectures**
- **Hayperparameters**
- **Future Work**


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


**DQN Architecture**

Both DQ-Networks (local and target) consist of 5 fully-connected layers ( 4 hidden layers, 1 output layers) each of hidden layers followed by a Relu activation function.

The number of neurons of the fully-connected layers are as follows:

- fc1 , number of neurons: 32,
- fc2 , number of neurons: 64,
- fc3 , number of neurons: 64,
- fc4 , number of neurons: 64,

- actions , number of neurons: 4,

**Dueling DQN Architecture**

As you can see from the picture above, the only difference between DQN and Dueling architecture is that Dueling DQN has two output layers, one called the advantage layer and the other called the value layer.

`Advantage function` captures how better an action is compared to the others at a given state, while as we know the `value function` captures how good it is to be at this state and The whole idea behind Dueling Q Networks relies on the representation of the Q function as a sum of the Value and the advantage function. We simply have two networks to learn each part of the sum and then we aggregate their outputs.


The number of neurons of the fully-connected layers are as follows:

- fc1 , number of neurons: 32,
- fc2 , number of neurons: 64,
- fc3 , number of neurons: 64,
- fc4 , number of neurons: 64,

- advantage , number of neurons: 4,
- value , number of neurons: 1,


### Hayperparameters

we know that Hyperparameters play an important role in deep learning, and it is a challenge to find the right values of hyperparameters. 
Here i give you a brief overview hyparameter, which i used in this project:

- learning rate    : 4e-5 
- number of layers : 5 - 6 Layers
- number of neurons: 4,32,64
- mini-batch size  : 64
- discount factor  : 0.99  
- TAU              : 1e-3
- epsilon start    : 1.0
- epsilon decay    : 0.99

### Results

| Original DQN | Double DQN | Dueling DQN | Dueling & Prioratized experience replay |
| ---------- | ---------- | ---------- | ---------- |
|![DQN][result]|![Double][result1] | ![Dueling][result2] | ![PER][result3] | 

### Future Work

While working on this project I dealt with many techniques that can be used to improve the Deep Q-Network. Some of these techniques I have already used in this project, and there are other methods like :

	- multi-level bootstrap targets (A3C) 
	- Distribution DQN 
	- Noisy DQN  
	- Rainbow: combaination of these sex different techniques.

I also accept Udacity challenge regarding navigation-pixels project, where the input is an **84x84 RGB image** instead of **state** as vector with 37 values. 
