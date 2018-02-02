# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset
import numpy as np


# Switch attractor dataset
class SwitchAttractorDataset(Dataset):
    """
    Generate a dataset where the reservoir must switch
    between two attractors.
    """

    # Constructor
    def __init__(self, sample_len, n_samples, seed=None):
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

        # Init seed if needed
        if seed is not None:
            torch.manual_seed(seed)
        # end if

        # Generate data set
        self.inputs, self.outputs = self._generate()
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
        return self.inputs[idx], self.outputs[idx]
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
        inputs = list()
        outputs = list()

        # Generate each sample
        for i in range(self.n_samples):
            # Start end stop
            start = np.random.randint(0, self.sample_len)
            stop = np.random.randint(start, start + self.sample_len / 2)

            # Limits
            if stop >= self.sample_len:
                stop = self.sample_len - 1
            # end if

            # Sample tensor
            inp = torch.zeros(self.sample_len, 1)
            out = torch.zeros(self.sample_len)

            # Set inputs
            inp[start, 0] = 1.0
            inp[stop] = 1.0

            # Set outputs
            out[start:stop] = 1.0

            # Add
            inputs.append(inp)
            outputs.append(out)
        # end for

        return inputs, outputs
    # end _generate

# end SwitchAttractorDataset
