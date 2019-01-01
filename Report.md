# Introduction
A Deep Deterministic Policy Gradient (DPDG) algorithm to solve Unity ML-agents Tennis environment.

Use Tenis.ipynb to train the agent. 

For each episode during training:

• Each agent collects an observation from the environment and takes an action using the policy, observes the reward and the next state.

• Stores the experience tuple SARS' in replay memory.
 
• Select a small bunch of tuples from memory randomly and learn from it.

# Deep deterministic Policy Gradient Agent
ddpg_agent.py has 3 main classes: 

- **Agent**, with parameters state_size, action_size, and a seed for random number generation in PyTorch.
- **ReplayBuffer**, initialized with parameters action_size, BUFFER_SIZE, BATCH_SIZE, and seed.
- **OUNoise** with inputs size, seed, mu, theta, sigma.


## Agent
Four neural networks are initialized with the Agent. Basically, two networks with two instances: an Actor and a Critic network, with two versions (target and local). The critic network estimates the value function of policy pi, V(pi) using the TD estimate. The output of the critic is used to train the actor network, which takes in a state, and outputs a distribution over possible actions.

The algorithm goes like this:
- Input the current state into the actor and get the action to take in that state. Observe next state and reward, to get an experience tuple s a r s'
- Then, using the TD estimate, which is the reward R plus the critic's estimate for s', the critic network is trained.
- Next, to calculate the advantage r+yV(s') - V(s), we also use the critic network.
- Finally, the actor is trained using the calculated advantage as a baseline.

Thus, the critic network learns the optimal action for every given state, and then passes that optimal action to the actor network which uses it to estimate the policy.
We use two networks for each of the two to control the update of the weights. 
Learning takes place at every UPDATE_EVERY timesteps: the weights are transferred between the target and local versions.

### Main functions

**step(self, state, action, reward, next_state, done):** Saves the experiences of both agents in the replay memory buffer, and calls the learn memory function at every UPDATE_STEPS time frames. The number of updates applied is defined in n_updates. 

**act (self, state, eps):**
Selects an action for a given state following the policy encoded by the NN function approximator. The architecture of this network is defined in the model.py file. Main steps:
1) Transformation of the current state from numpy to torch 
2) Forward pass on the actor_local. 
3) Data is moved to a cpu and to numpy
4) Noise is added
5) Clipping between -1 and 1.


**learn(self, experiences, gamma):**
Updates the Actor and Critic network’s weights given a batch of experience tuples.
1) In the update critic section of the function, we first get the max predicted Q values (for next states) from the critic_target and actor_target models, and compute Q targets for current states. Then, we get the expected Q values from the critic_local model, compute the loss and minimize the loss using gradient descent. Note that we use gradient clipping.

2) In the update actor section, we compute the loss of the Actor, and get the predicted actions.
3) The function soft_update is called in the end to update the target networks.
4) Epsilon is updated to decrease over time until reaching a minimum. This is to update noise in the act function

**soft update (local_model, target_model, tau):**
Grabs all of the target_model and the local_model parameters (in the zip command), and copies a mixture of both (defined by Tau) into the target_param.
The target network receives updates that are a combination of the local (most up to date network) and the target (itself). In general, it will be a very large chunk of itself and a very small chunk of the other network.

## Replay Buffer
The replay buffer class retains the end most recent experience tuples. 
Buffer is implemented with a python deque. Note that we do not clear out the memory after each episode, which enables to recall and build batches of experience from across episodes.
Given that maxlen is specified to BATCH_SIZE, the buffer is bounded. Once full, when new items are added, a corresponding number of items are discarded from the opposite end.

## OUNoise
Implementation of stochastic noise.


# Neural Network Architecture
Two classes are instantiated in the model.py file for Actor and Critic networks.

**The actor network:**
Neural Network that estimates the optimal policy. 
Built in PyTorch (nn package). 
The architecture includes an input layer (of size = state size), two fully connected hidden layers of 400 and 300 units and an output layer (size = action size).
RELU (Regularized linear units) activation is applied in the forward function for the first two layers. Note that on the output side, given the continuous space we use the Tahn activation function.

**The critic network:**
Approximates the value function. It is defined by subclassing torch.nn.Module. 
The architecture includes an input layer (of size = state size), one fully connected hidden layer of 400 units, a second hidden layer of 300 + action size units and an output layer (size = action size).
RELU activation is applied in the forward function. Note that in the forward pass, the action is being concatenated to get the value of the specific state-action pair.


# Chosen hyperparameters

- BUFFER_SIZE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=  int(5e5)  &nbsp;&nbsp;&nbsp;#replay buffer size
- BATCH_SIZE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=  1024 &nbsp;&nbsp;&nbsp;#minibatch size
- GAMMA &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=  0.99 &nbsp;&nbsp;&nbsp;#discount factor
- TAU   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=  1e-3 &nbsp;&nbsp;&nbsp;#for soft update of target parameters
- LR_ACTOR &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=  1e-3 &nbsp;&nbsp;&nbsp;#learning rate of the actor
- LR_CRITIC &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=  1e-3 &nbsp;&nbsp;&nbsp;#learning rate of the critic 
- WEIGHT_DECAY &nbsp;=  0 &nbsp;&nbsp;&nbsp;#L2 weight decay



# Training protocol
After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The agent was trained until an average score of +0.5 over 100 consecutive episodes was reached. 


![alt text](/output2.png?raw=true "Title")

Figure shows average reward for each episode both agents. The goal of one hundred consecutive episodes with scores above 0.5 was achieved at episode xxx.



# Ideas for future work
- Try different neural network architectures for both Actor and Critic networks. In particular, add hidden layers to improve the policy and value function estimation.
- Try other algorithms such as PPO or AC2. 
- Train the agent using raw pixels.
- Try out different hyperparameters. Due to the long training time, it was not possible to test different variations. 


