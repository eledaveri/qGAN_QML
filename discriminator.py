# discriminator.py — Classic Discriminator (PyTorch)
# Architecture: input(50) → LeakyReLU → hidden(20) → LeakyReLU → output(1) → Sigmoid
# The integer samples are one-hot encoded to DISCRIMINATOR_INPUT_DIM dimensions.
# The model and tensors are moved to the device defined in config.py (GPU/CPU).

import numpy as np
import torch
import torch.nn as nn

from config import DISCRIMINATOR_INPUT_DIM, DEVICE


class Discriminator(nn.Module):
    """
    Classic Discriminator described in the paper.

    Architecture (paper, section Methods)
    ------
    Linear(50→50) → LeakyReLU(0.2) → Linear(50→20) → LeakyReLU(0.2)
    → Linear(20→1) → Sigmoid

    The input is a one-hot vector with DISCRIMINATOR_INPUT_DIM dimensions
    (padded with zeros if n_values < DISCRIMINATOR_INPUT_DIM).
    The model is automatically moved to DEVICE (GPU if available).
    """

    def __init__(self, input_dim: int = DISCRIMINATOR_INPUT_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.LeakyReLU(0.2),
            nn.Linear(50, 20),
            nn.LeakyReLU(0.2),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )
        self.to(DEVICE)  # move weights to GPU if available

    def encode(self, samples: np.ndarray, n_values: int) -> torch.Tensor:
        """
        Convert an array of integers to a batch of one-hot vectors
        already positioned on DEVICE.

        Parameters
        ----------
        samples  : array of integers in [0, n_values)
        n_values : number of discrete values of the generator

        Returns
        -------
        FloatTensor di shape (len(samples), self.input_dim) on DEVICE
        """
        x = np.zeros((len(samples), self.input_dim), dtype=np.float32)
        for i, s in enumerate(samples):
            if 0 <= s < n_values:
                x[i, s] = 1.0
        return torch.from_numpy(x).to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)