# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset
import collections


# Mackey Glass dataset
class MackeyGlassDataset(Dataset):
    """
    Mackey Glass dataset
    """

    # Constructor
    def __init__(self, sample_len, n_samples, tau=17, seed=None):
        """
        Constructor
        :param sample_len: Length of the time-series in time steps.
        :param n_samples: Number of samples to generate.
        :param tau: Delay of the MG with commonly used value of tau=17 (mild chaos) and tau=30 is moderate chaos.
        :param seed: Seed of random number generator.
        """
        # Properties
        self.sample_len = sample_len
        self.n_samples = n_samples
        self.tau = tau
        self.delta_t = 10
        self.timeseries = 1.2
        self.history_len = tau * self.delta_t

        # Init seed if needed
        if seed is not None:
            torch.manual_seed(seed)
        # end if
    # end __init__

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return self.n_samples
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        # History
        history = collections.deque(1.2 * torch.ones(self.history_len) + 0.2 * (torch.rand(self.history_len) - 0.5))

        # Preallocate tensor for time-serie
        inp = torch.zeros(self.sample_len, 1)

        # For each time step
        for timestep in range(self.sample_len):
            for _ in range(self.delta_t):
                xtau = history.popleft()
                history.append(self.timeseries)
                self.timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - 0.1 * history[-1]) / self.delta_t
            # end for
            inp[timestep] = self.timeseries
        # end for

        # Inputs
        inputs = torch.tan(inp - 1)

        # Squash timeseries through tanh
        return inputs[:-1], inputs[1:]
    # end __getitem__

# end MackeyGlassDataset
