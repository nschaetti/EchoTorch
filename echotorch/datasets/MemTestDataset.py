# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset


# Generates a series of input timeseries and delayed versions as outputs.
class MemTestDataset(Dataset):
    """
    Generates a series of input timeseries and delayed versions as outputs.
    Delay is given in number of timesteps. Can be used to empirically measure the
    memory capacity of a system.
    """

    # Constructor
    def __init__(self, sample_len, n_samples, n_delays=10, seed=None):
        """
        Constructor
        :param sample_len: Length of the time-series in time steps.
        :param n_samples: Number of samples to generate.
        :param n_delays: Number of step to delay
        :param seed: Seed of random number generator.
        """
        # Properties
        self.sample_len = sample_len
        self.n_samples = n_samples
        self.n_delays = n_delays

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
        inputs = (torch.rand(self.sample_len, 1) - 0.5) * 1.6
        outputs = torch.zeros(self.sample_len, self.n_delays)
        for k in range(self.n_delays):
            outputs[:, k:k+1] = torch.cat((torch.zeros(k + 1, 1), inputs[:-k - 1, :]), dim=0)
        # end for
        return inputs, outputs
    # end __getitem__

# end MemTestDataset
