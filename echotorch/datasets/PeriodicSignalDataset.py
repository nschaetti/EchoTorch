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
    def __init__(self, sample_len, period, n_samples, height=1.8, start=0, dtype=torch.float32):
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
        self.height = height
        self.dtype = dtype

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

        # Pattern
        maxVal = torch.max(self.period)
        minVal = torch.min(self.period)
        rp = self.height * (self.period - minVal) / (maxVal - minVal) - (self.height / 2.0)
        p_length = rp.size(0)

        # For each sample
        for i in range(self.n_samples):
            # Tensor
            sample = torch.zeros(self.sample_len, 1, dtype=self.dtype)

            # Timestep
            for t in range(self.sample_len):
                sample[t, 0] = rp[(t + self.start) % p_length]
            # end for

            # Append
            samples.append(sample)
        # end for

        return samples
    # end _generate

# end PeriodicSignalDataset
