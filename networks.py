import torch
import torch.nn as nn

from typing import Tuple


class Network(nn.Module):

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