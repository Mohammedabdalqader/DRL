[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[result]: Navigation/results/DQN.png "DQN"
[result1]: Navigation/results/Double.png "Double"
[result2]: Navigation/results/Dueling.png "Dueling"
[result3]: Navigation/results/PER.png "PER"

# Project 1: Navigation

After learning several reinforcement learning algorithms and techniques during Udacity DLR Nanodegree, I worked on Navigation Project and I will share my implementations for this project with you. 

For this project I have trained my Agent with 4 different methods:
- **`DQN`**
- **`Double DQN`**
- **`Dueling DQN`**
- **`Dueling DQN with Prioratized experience replay`**

You will find all these implementations and checkpoints in the "Navigation" folder

### Project description
For this project, i will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drl python=3.6
	source activate drl
	```
	- __Windows__: 
	```bash
	conda create --name drl python=3.6 
	activate drl
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the Repository, and navigate to the DRL/Navigation/  folder
    ```bash
    git clone https://github.com/Mohammedabdalqader/DRL.git
    cd DRL/Project-1_Navigation/Navigation
    ```
4. Set up your Python environment. 
    you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required for this project.
    ```
    cd ../python/
    pip install .
    ```

5. Download the Unity Environment.

    For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can  download it from one of the links below. You need only select the environment that matches your operating system:

    * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    Then, place the file in the Navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

    (For Windows (64-bit) users) I have provided Unity enviroment for this Project for you, so you don't need to download it by yourself 


6. Create an IPython kernel for the drl (deep reinforcement learning) environment.
    ```bash
    python -m ipykernel install --user --name drl --display-name "drl"
    ```

7. Before running code in a notebook, change the kernel to match the `drl` environment by using the drop-down `Kernel` menu. 


### Instructions

To start training your own agent, all you have to do is to follow the instructions included in each of these 4 Jupyter notebooks:

- 01.Navigation-DQN.ipynb
- 02.Navigation-Double_DQN.ipynb
- 03.Navigation-Dueling_DQN.ipynb
- 04.Navigation-Dueling_DQN-&-Prioritized_experience_replay.ipynb


# Results
| Original DQN | Double DQN | Dueling DQN | Dueling & Prioratized experience replay |
| ---------- | ---------- | ---------- | ---------- |
|![DQN][result]|![Double][result1] | ![Dueling][result2] | ![PER][result3] | 
# Future Work

While working on this project I dealt with many techniques that can be used to improve the Deep Q-Network. Some of these techniques I have already used in this project, and there are other methods like :

	- multi-level bootstrap targets (A3C) 
	- Distribution DQN 
	- Noisy DQN  
	- Rainbow: combaination of these sex different techniques.
	
After implemnting these methods i will update this Repo and share the results :smiley:	

I also accept Udacity challenge regarding navigation-pixels project, where the input is an **84x84 RGB image** instead of **state** as vector with 37 values. 

	

