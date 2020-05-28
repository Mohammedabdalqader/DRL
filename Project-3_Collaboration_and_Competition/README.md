[//]: # (Image References)

[trained_agents]: Collaboration_and_Competition/images/traind_agents.gif "TA"
[result]: Collaboration_and_Competition/images/maddpg.png "MADDPG"


# Project 3: Collaboration and Competition

For this project I have trained my Agents with MADDPG (Multi-Agent Deep Deterministic Policy Gradient) algorithm in a competitive environment where the goal of each agent is to maximize its own returns.

You will find my implementation and checkpoints in the "Collaboration_and_Competition" folder

### Project description
In this project I have worked with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![TA][trained_agents]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

	* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
	* This yields a single score for each episode.


### Getting Started

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name MADRL python=3.6
	source activate MADRL
	```
	- __Windows__: 
	```bash
	conda create --name MADRL python=3.6 
	activate MADRL
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the Repository, and navigate to the DRL/Project-3_Collaboration_and_Competition/Collaboration_and_Competition  folder
    ```bash
    git clone https://github.com/Mohammedabdalqader/DRL.git
    cd DRL/Project-3_Collaboration_and_Competition/Collaboration_and_Competition
    ```
4. Set up your Python environment. 
    you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required for this project.
    ```
    cd ../python/
    pip install .
    ```

5. Download the Unity Environment.

    For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can  download it from one of the links below. You need only select the environment that matches your operating system:
	
    
	* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
	* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
	* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
	* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
		

		
    Then, place the file in the Collaboration_and_Competition/ folder in the DRL GitHub repository, and unzip (or decompress) the file.
		
    (For Windows (64-bit) users) I have provided Unity enviroment for this Project for you, so you don't need to download it by yourself 


6. Create an IPython kernel for the drl (deep reinforcement learning) environment.
    ```bash
    python -m ipykernel install --user --name MADRL --display-name "MADRL"
    ```

7. Before running code in a notebook, change the kernel to match the `MADRL` environment by using the drop-down `Kernel` menu. 


### Instructions

To start training your own agent, all you have to do is to follow the instructions included in this jupyter notebook Competitive_Multi-Agent.ipynb :

# Results
| MADDPG (Multi-Agent Deep Deterministic Policy Gradient)|
| ---------- |
|![MADDPG][result]|

# Future Work

After 2 months with the excellent knowledge that this course has given us, I can say that I have taken a big step towards mastering this area. I am able to implement different algorithms and to select a suitable one for each problem.
In this project i have a chieved a very good result, in less than 200 episodes the target average reward achieved (> 0.50) and in 250 episode the average reward was 1.338 :muscle:. but I wonder if the performance will be better if I use prioritized experience replay? So I will work on it, and if it gives a better result, I will share the results with you :grinning:
