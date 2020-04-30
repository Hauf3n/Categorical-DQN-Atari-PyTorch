import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class C51_DQN(nn.Module):
    # nature paper architecture
    
    def __init__(self, in_channels, num_actions, num_atoms):
        super().__init__()
        
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        
        network = [
            torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7,512),
            nn.ReLU(),
            nn.Linear(512, num_actions*num_atoms)
        ]
        
        self.network = nn.Sequential(*network)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        out = self.network(x)
        out = torch.reshape(out, (batch_size, self.num_actions, self.num_atoms))
        value_distribution = F.softmax(out, dim=2)
        
        return value_distribution