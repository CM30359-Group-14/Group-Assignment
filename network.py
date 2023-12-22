import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional

#very basic NN

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
    
    def __init__(self, obs_dims, action_dims):
        print(obs_dims)
        super(Network, self).__init__()
        self.layer1 = nn.Linear(obs_dims, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, action_dims)
        
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        
        activation1 = functional.relu(self.layer1(obs))
        activation2 = functional.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        return output