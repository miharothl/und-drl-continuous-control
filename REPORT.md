[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135608-be87357e-7d12-11e8-8eca-e6d5fabdba6b.gif "Bipedal Walker"
[image3]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"

# Goal

The goal of the project is to train an agent and control 20 robot arms to reach the randomly circulating spheres, circulating around the robot.


![Trained Agent][image1]

We take into account the presence of many agents which need to get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

# Approach

I started with the base DDPG [2] alghoritm provided by [Udacity](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal)
[1], which can train a bipedal walker.
 

![Bipedal Walker][image2]

I adjusted the code to the Unity Reacher environment, and the code agent was able to achieve 30+ scores over 100 consecutive episodes.
I tried to change many different hyperparameters and followed the procedure outlined by Udacity.

1. Adjust to code to use multiple agents, that during training use shared expericen replay buffer
2. Implemented less agressive training stragegy, updating the network 20 times avery 10 timesteps.

The steps that I followed to solve this environment

1. Evaluate the state and action space of the environment
2. Establish a baseline using a random action policy
3. Implement the learning algorithm
4. Run experiments and select the best agent

## Evaluate State & Action Space of the Environment

The state-space has 33 dimensions corresponding to position, rotation, velocity, and angular velocities of the two arm rigid bodies.
Action-space is continuous, it has 4 dimensions corresponding to torque applicable to two joints.

## Establish Baseline Using Random Action Policy

Before starting the deep reinforcement learning process its good to understand the environment. Controlling the 
robots with agent where actions are randomly selected achives scores averaging 0 over 100 consecutive episodes.
 
## Implement Learning Algorithm

The
[agent](https://github.com/miharothl/...ddpg_agent.py)
and 
[environment](https://.../unity_multiple_env.py)
are created according to the provided
[configuration](https://.../configuration.py)
.
[Recorder](https://.../recorder.py)
records the experiment and store the results for later
[analysis](https://.../analysis.ipynb)
.

The agent interacts with the environment in the
[training loop](https://.../master_trainer.py)
.
In the exploration phase (higher *Epsilon*) of the training
agent's actions are mostly random, greated using 
[Ornstein-Uhlenbeck noise generator] (https://.../ou_noise.py)
. Actions, environment states, dones, and rewards tuples, are stored in the experience
replay buffer. The *Buffer Size* parameter determines the size of the buffer.

DDPG is using actor and critic neural networks. Both have current and target model with identical architecture are used to stabilise the DDPG learning process.
During the learning process, weights of the target network are fixed (or updated more slowly based on parameter *Tau*).

Learning is performed *Num Updates* times on every *Update Every* steps, when *Batch Size* of actions, states, dones and rewards tuples are
sampled from the randomly sampled from 
[replay buffer](https://.../replay_buffer.py)
.

During the exploitation phase of the training (lower *Epsilon*) the the noise added to the actions is proportionally sampled down (*epsilon end*)
and mostly based on the estimated Q values calculated by the current neural network.


## Run Experiments and Select Best Agent

[Training](https://..../continous-control.ipynb)
is done using the epochs, consisting of training episodes where epsilon greedy agent is used,
and validation episodes using only actions predicted by the trained agent. I used the following training hyper parameters:

|Hyper Parameter            |Value                 |
|:---                       |:---                  |
|Max Steps                  |300000 (300 episodes) |
|Max Episode Steps          |1000                  |
|Evaluation Frequency       |10000  (10 episodes)  |
|Evaluation Steps           |2000   (2 episodes)   |
|Epsilon Start              |1.                    |
|Epsilon End                |0.1                   |
|Epsilon Decay              |0.97                  |
|Batch Size                 |128                   |
|Update Every               |10                    |
|Num Updates                |20                    |
|Learning Rate Actor        |0.0001                |
|Learning Rate Critic       |0.0003                |
|Tau                        |0.001                 |
|Gamma                      |0.99                  |
|Actor Hidden Layers Units  |[256, 128]            |
|Critic Hidden Layers Units |[256, 128]            |
|Buffer Size                |100000                |
|Use Prioritized Replay     | False                |
 
The environment is solved in epcch x after playing 30 episodes. The trained agent achieves an average score of 38.51 over 100 episodes.

# Future Work

Deep reinforcement learning is a fascinating and exciting topic. I'll continue to improve my reinforcement learning
laboratory by solving the crawler.

![Crawler][image3]

# References
  - [1] [Udacity](https://github.com/udacity/deep-reinforcement-learning)
  - [2] [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
