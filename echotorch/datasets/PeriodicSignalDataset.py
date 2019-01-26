# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset
import numpy as np


# Periodic signal timeseries
class PeriodicSignalDataset(Dataset):
    """
    Create simple periodic signal timeseries
    """

    # Constructor
    def __init__(self, sample_len, period, n_samples, start=0):
        """
        Constructor
        :param sample_len: Sample's length
        :param period:
        """
        # Properties
        self.sample_len = sample_len
        self.n_samples = n_samples
        self.period = period
        self.start = start

        # Period length
        if type(period) is list:
            self.period_length = len(period)
        elif type(period) is np.array or type(period) is torch.FloatTensor:
            self.period_length = period.shape[0]
        # end if

        # Generate data set
        self.outputs = self._generate()
    # end __init__

    #############################################
    # OVERRIDE
    #############################################

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
        return self.outputs[idx]
    # end __getitem__

    ##############################################
    # PRIVATE
    ##############################################

    # Generate
    def _generate(self):
        """
        Generate dataset
        :return:
        """
        # List of samples
        samples = list()

        # For each sample
        for i in range(self.n_samples):
            # Tensor
            period_tensor = torch.FloatTensor(self.period)
            sample = period_tensor.repeat(int(self.sample_len // self.period_length) + 1)

            # Start
            if type(self.start) is list:
                start = self.start[i]
            else:
                start = self.start
            # end if

            # Append
            samples.append(sample[start:start+self.sample_len].unsqueeze(-1))
        # end for

        return samples
    # end _generate

# end PeriodicSignalDataset
