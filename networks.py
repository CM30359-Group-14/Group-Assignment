import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from layers import NoisyLinear


class Network(nn.Module):
    """
    Class representing a feed-forward multi-layered percepton.
    """

    def __init__(self, in_dim: int, out_dim: int):
        """
        Instantiates a three-layer feed-forward neural network.
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the neural network.
        """
        return self.layers(x)
    

class CNNNetwork(nn.Module):

    def __init__(self, input_shape: Tuple, out_dim: int):
        super().__init__()

        self.input_shape = input_shape
        self.out_dim = out_dim

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten() # Flatten the conv layer ouput to feed it to the fully-connected layer
        )

        # Compute the flattened size after the conv layers to properly initialise the FC layer.
        with torch.no_grad():
            self.feature_size = self.features(torch.zeros(1, *input_shape)).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)

        return self.fc(features)
    

class NoisyNetwork(nn.Module):
    """
    Class representing a Noisy Neural Network for exploration.
    """

    def __init__(self, in_dim: int, out_dim: int):
        """
        Instantiates a NoisyNetwork.
        """
        super().__init__()

        self.features = nn.Linear(in_dim, 128)
        self.noisy_layer1 = NoisyLinear(128, 128)
        self.noisy_layer2 = NoisyLinear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the neural network.
        """
        features = F.relu(self.features(x))
        hidden = F.relu(self.noisy_layer1(features))

        return self.noisy_layer2(hidden)
    
    def reset_noise(self):
        """
        Resets all noisy layers.
        """
        self.noisy_layer1.reset_noise()
        self.noisy_layer2.reset_noise()


class CategoricalNetwork(nn.Module):
    """
    Class representing a Categorical Neural Network.
    """

    def __init__(self, in_dim: int, out_dim: int, atom_size: int, support: torch.Tensor):
        """
        Instantiates a CategoricalNetwork.
        """
        super().__init__()

        self.support = support
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.atom_size = atom_size

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim * atom_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the neural network.
        """
        dist = self.dist(x)

        return torch.sum(dist * self.support, dim=2)
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gets a distribution over atoms.
        """
        q_atoms = self.layers(x).view(-1, self.out_dim, self.atom_size)
        dist = F.softmax(q_atoms, dim=-1)
        # To avoid NaNs
        dist = dist.clamp(min=1e-3)

        return dist