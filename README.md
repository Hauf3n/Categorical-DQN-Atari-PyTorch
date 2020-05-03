# Categorical_DQN-Atari-PyTorch
Implementation of [Categorical DQN (C51)](https://arxiv.org/abs/1707.06887) by Bellemare et al.<br />

Applied to the gym Seaquest, Breakout, Pong and SpaceInvaders environment. *NoFrameskip-v4

# Results - Seaquest - Random Run
 ![games](https://github.com/Hauf3n/Categorical_DQN-Atari-PyTorch/blob/master/media/seaquest_37k.gif)
 [Youtube](https://youtu.be/siPcgY4ikk0)<br /><br />
 
 # Training
 
 My C51 algorithm is a bit more unstable than the paper results but still good, especially Seaquest.<br />
 
 Training: Seaquest <br />
 ![seaq](https://github.com/Hauf3n/Categorical_DQN-Atari-PyTorch/blob/master/media/seaquest_plot.png)<br />
 
 Training: Breakout <br />
 ![breakout](https://github.com/Hauf3n/Categorical_DQN-Atari-PyTorch/blob/master/media/breakout_plot.png)<br />
 
 Training: Pong <br />
 ![pong](https://github.com/Hauf3n/Categorical_DQN-Atari-PyTorch/blob/master/media/pong_plot.png)<br />
 
 Training: SpaceInvaders <br />
 sparse rewards at ~(600-800) return. Often only one/two, fast moving targets left. Hard to optimize!
 ![space](https://github.com/Hauf3n/Categorical_DQN-Atari-PyTorch/blob/master/media/spaceinvaders_plot.png)<br />
