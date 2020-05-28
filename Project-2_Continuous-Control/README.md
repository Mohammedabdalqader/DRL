[//]: # (Image References)

[random_agents]: Continuous-Control/images/random_agents.gif "RA"
[trained_agents]: Continuous-Control/images/trained_agents.gif "TA"
[result]: Continuous-Control/images/d4pg.png "D4PG"


# Project 2: Continuous Control

For this project I have trained my Agent with Distributed distributional deep deterministic policy gradients (D4PG) algorithm:

You will find my implementation and checkpoints in the "Continuous-Control" folder

### Project description
In this project I have worked with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

| Random agents| trained agents | 
| ---------- | ---------- |
|![RA][random_agents]|![TA][trained_agents] |

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 
Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers,
corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

In this repo I have provided you with two different environments

**`Option 1`**: First Version
The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

**`Option 2`**: Second Version
The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
* This yields an average score for each episode (where the average is over all 20 agents).

### Getting Started

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name continuous-control python=3.6
	source activate continuous-control
	```
	- __Windows__: 
	```bash
	conda create --name continuous-control python=3.6 
	activate continuous-control
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the Repository, and navigate to the DRL/Navigation/  folder
    ```bash
    git clone https://github.com/Mohammedabdalqader/DRL.git
    cd DRL/Project-2_Continuous-Control/Continuous-Control
    ```
4. Set up your Python environment. 
    you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required for this project.
    ```
    cd ../python/
    pip install .
    ```

5. Download the Unity Environment.

    For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can  download it from one of the links below. You need only select the environment that matches your operating system:
	
    **`Version 1: One (1) Agent`**:
    
	* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
	* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
	* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
	* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
		

    **`Version 2: Twenty (20) Agents`**:
	
	* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
	* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
	* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
	* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
		
    Then, place the file in the Continuous-Control/ folder in the DRL GitHub repository, and unzip (or decompress) the file.
		
    (For Windows (64-bit) users) I have provided **`Version 2`** Unity enviroment for this Project for you, so you don't need to download it by yourself 


6. Create an IPython kernel for the drl (deep reinforcement learning) environment.
    ```bash
    python -m ipykernel install --user --name continuous-control --display-name "continuous-control"
    ```

7. Before running code in a notebook, change the kernel to match the `continuous-control` environment by using the drop-down `Kernel` menu. 


### Instructions

To start training your own agent, all you have to do is to follow the instructions included in this jupyter notebook Continuous_Control.ipynb :

# Results
| Distributed distributional deep deterministic policy gradients (D4PG) |
| ---------- |
|![D4PG][result]|

# Future Work

While working on this project, I had to invest too much time in research to find the right algorithms for such a problem. There were many options available to me, and this was a challenge for me, and from here my journey began.

There is really a very useful [repo](https://github.com/ShangtongZhang/DeepRL) that describes and implements different algorithms that work very well for such a problem with continuous action space. Thanks to this repo and other sources, I was able to understand some algorithms correctly, including the DDPG, D4PG, PPO, A2C, and A3C algorithms, and I was able to implement some of these algorithms to solve my problem.

Here are some Ideas for improvement:

* Implementing TRPO, PPO, A3C, A2C algorithms:

  It is worthwhile to implement all these algorithms, so I will work on it in the next days and see which of these algorithms converges   faster. 

* Adjusting the Hyperparameters:

  The more important step I can also take to improve the results and solve the problem with 100 episodes or even < 100 is to adjust the   hyperparameters. 

* Using prioritized experience replay and N-step techniques:

  As mentioned in this paper https://openreview.net/forum?id=SyZipzbCb using these techniques with D4PG could potentially lead to better  results

