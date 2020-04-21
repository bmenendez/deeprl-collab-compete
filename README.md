# Deep Reinforcement Learning project: tennis
## Project details

In this project, two agents are trained to solve an environment in which they must control rackets to bounce a ball over a net.

![tennis environment](images/tennis.png "Tennis environment")

If an agent hits the ball over the net, it receives a reward of +0.1.   If an agent lets a ball hit the ground or hits the ball out of bounds,  it receives a reward of -0.01.  Thus, the goal of each agent is to keep  the ball in play.

The observation space consists of 8 variables corresponding to the  position and velocity of the ball and racket. Each agent receives its  own, local observation.  Two continuous actions are available,  corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your  agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received  (without discounting), to get a score for each agent. This yields 2  (potentially different) scores. We then take the maximum of these 2  scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Getting started

1. Follow the instructions given [here](https://github.com/udacity/deep-reinforcement-learning#dependencies) to install all the dependencies.

2. Download the environment for your OS:

   * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip).
   * Mac OS: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip).
   * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip).
   * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip).
3. Place the file in the root of the folder and unzip it.

4. Run it! Consider to change the cell at point 2 of the notebook to match with your folder `env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86_64')`.

## Instructions

If you want to train the agents, run the cells from the point 1 to the point 6 at [MADDPG.ipynb](MADDPG.ipynb). If you just want to execute the trained agents, run the cells from the point 1 to the point 4 plus the 7 at the notebook, since they load the weights of the networks ([checkpoint_actor_0.pth](checkpoint_actor_0.pth) for the actor and [checkpoint_critic_0.pth](checkpoint_critic_0.pth) for the critic of the first agent, [checkpoint_actor_1.pth](checkpoint_actor_1.pth) and [checkpoint_critic_1.pth](checkpoint_critic_1.pth) for the actor and the critic of the second agent) for the trained agents. After running whatever you want, you can close the environment by running the cell at point 8.

Description of the implementation is in [Report.md](Report.md), but for more technical details, see the code at the notebook provided before.