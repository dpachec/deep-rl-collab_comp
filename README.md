# Project Details

In this project, we will train 2 agents to play tenis in the Tennis enviroment of Unity ML-agents. Our agents will store observations of the environment in a continuous vector of 8 values corresponding to position and velocity of the ball and racket. Each agent will have 2 possible actions available at each time step, corresponding to movement toward (or away from) the net, and jumping.

If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The task is episodic, and in order to solve the environment, our agent must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents)



# Getting Started

The project requires Python 3.6 or higher with the libraries unityagents, numpy, PyTorch.

You will also need the Unity Tennis environment, which can be downloaded [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip).


# Installation
1) Clone the repo (Anaconda):
```
git clone https://github.com/dpachec/deep-rl-collab-comp.git
```

2) Install Dependencies using Anaconda
```
conda create --name drlnd_collab-comp python=3.6
source activate drlnd_collab-comp
conda install -y python.app
conda install -y pytorch -c pytorch
pip install unityagents
```

# Instructions

The first cell of Tenis.ipynb will train 2 agents. The weights of the Actor and Critic neural network implementations will be saved into "checkpoint_actor.pth" and "checkpoint_critic.pth".

To observe the behavior of the trained agent, load the weights from the files and run the simulation in the second cell.







