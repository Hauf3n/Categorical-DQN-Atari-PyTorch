# Categorical-DQN-Atari-PyTorch
Implementation of [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887) by Bellemare et al.<br />
Algorithm: Categorical DQN (C51)<br />

Applied to the gym Seaquest, Breakout, Pong and SpaceInvaders environment. *NoFrameskip-v4

# Idea
Replace the output of a Q-network (expected return) with a distribution over returns.<br />
However, C51 will kind of compute the expected return over all defined returns.<br /><br />
The core idea is that the C51-Q-network can have different representations for each outcome,<br />
because of the distribution property. It means that the last layer of the network does not need<br /> 
to encode 2 observations ,which have the same expected return, with the same activation pattern.

# Results - Seaquest - Random Run
 ![games](https://github.com/Hauf3n/Categorical_DQN-Atari-PyTorch/blob/master/media/seaquest_37k.gif)
 [Youtube](https://youtu.be/siPcgY4ikk0)<br /><br />
 
 # Training
 
 My C51 algorithm is a bit more unstable than the paper results but still good, especially Seaquest.<br />
 Improved the algorihm speed by vectorizing the for loop.
 
 Training: Seaquest <br />
 ![seaq](https://github.com/Hauf3n/Categorical_DQN-Atari-PyTorch/blob/master/media/seaquest_plot.png)<br />
 
 Training: Breakout <br />
 ![breakout](https://github.com/Hauf3n/Categorical_DQN-Atari-PyTorch/blob/master/media/breakout_plot.png)<br />
 
 Training: Pong <br />
 ![pong](https://github.com/Hauf3n/Categorical_DQN-Atari-PyTorch/blob/master/media/pong_plot.png)<br />
 
 Training: SpaceInvaders <br />
 sparse rewards at ~(600-800) return. Often only one/two, fast moving targets left. Hard to optimize!
 ![space](https://github.com/Hauf3n/Categorical_DQN-Atari-PyTorch/blob/master/media/spaceinvaders_plot.png)<br />
