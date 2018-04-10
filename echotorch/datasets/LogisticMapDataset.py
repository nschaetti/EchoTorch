# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset
import numpy as np


# Logistic Map dataset
class LogisticMapDataset(Dataset):
    """
    Logistic Map dataset
    """

    # Constructor
    def __init__(self, sample_len, n_samples, alpha=5, beta=11, gamma=13, c=3.6, b=0.13, seed=None):
        """
        Constructor
        :param sample_len:
        :param n_samples:
        :param alpha:
        :param beta:
        :param gamma:
        :param c:
        :param b:
        :param seed:
        """
        # Properties
        self.sample_len = sample_len
        self.n_samples = n_samples
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.c = c
        self.b = b
        self.p2 = np.pi * 2

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
        # Time and forces
        t = np.linspace(0, 1, self.sample_len, endpoint=0)
        dforce = np.sin(self.p2 * self.alpha * t) + np.sin(self.p2 * self.beta * t) + np.sin(self.p2 * self.gamma * t)

        # Series
        series = torch.zeros(self.sample_len, 1)
        series[0] = 0.6

        # Generate
        for i in range(1, self.sample_len):
            series[i] = self._logistic_map(series[i-1], self.c + self.b * dforce[i])
        # end for

        return series
    # end __getitem__

    #######################################
    # Private
    #######################################

    # Logistic map
    def _logistic_map(self, x, r):
        """
        Logistic map
        :param x:
        :param r:
        :return:
        """
        return r * x * (1-x)
    # end logistic_map

# end MackeyGlassDataset
