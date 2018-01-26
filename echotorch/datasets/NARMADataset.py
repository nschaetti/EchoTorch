# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset


# 10th order NARMA task
class NARMADataset(Dataset):
    """
    xth order NARMA task
    WARNING: this is an unstable dataset. There is a small chance the system becomes
    unstable, leading to an unusable dataset. It is better to use NARMA30 which
    where this problem happens less often.
    """

    # Constructor
    def __init__(self, sample_len, n_samples, system_order=10, seed=None):
        """
        Constructor
        :param sample_len: Length of the time-series in time steps.
        :param n_samples: Number of samples to generate.
        :param system_order: th order NARMA
        :param seed: Seed of random number generator.
        """
        # Properties
        self.sample_len = sample_len
        self.n_samples = n_samples
        self.system_order = system_order

        # System order
        self.parameters = torch.rand(4)
        if system_order == 10:
            self.parameters[0] = 0.3
            self.parameters[1] = 0.05
            self.parameters[2] = 9
            self.parameters[3] = 0.1
        else:
            self.parameters[0] = 0.2
            self.parameters[1] = 0.04
            self.parameters[2] = 29
            self.parameters[3] = 0.001
        # end if

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
        inputs = torch.rand(self.sample_len, 1) * 0.5
        outputs = torch.zeros(self.sample_len, 1)
        for k in range(self.system_order-1, self.sample_len - 1):
            outputs[k + 1] = self.parameters[0] * outputs[k] + self.parameters[1] * outputs[k] * torch.sum(
                outputs[k - (self.system_order - 1):k + 1]) + 1.5 * inputs[k - int(self.parameters[2])] * inputs[k] + \
                             self.parameters[3]
        # end for

        return inputs, outputs
    # end __getitem__

# end MackeyGlassDataset
