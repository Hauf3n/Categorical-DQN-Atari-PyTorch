import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Networks import C51_DQN

class C51_Agent(nn.Module):
    
    def __init__(self, in_channels, num_actions, num_atoms, value_support, epsilon):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.value_support = value_support
        self.eps = epsilon
        
        self.network = C51_DQN(in_channels, num_actions, num_atoms)
        
    def forward(self, x):
        value_distribution = self.network(x)
        return value_distribution
    
    def e_greedy(self, x):
        
        greedy = torch.rand(1)
        if self.eps < greedy:
            value_distribution = self.forward(x).detach()
            return self.greedy(value_distribution)
        else:
            return (torch.rand(1) * self.num_actions).type('torch.LongTensor')
        
    def greedy(self, value_distribution):
        # batchsize, actions, atoms
        
        Q_values = value_distribution @ self.value_support
        action = torch.argmax(Q_values, dim=1)

        return action
    
    def set_epsilon(self, epsilon):
        self.eps = epsilon