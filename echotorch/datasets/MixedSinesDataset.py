# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset
import math
import numpy as np


# Mixed sines dataset
class MixedSinesDataset(Dataset):
    """
    Mixed sines dataset
    """

    # Constructor
    def __init__(self, sample_len, n_samples, periods, amplitudes, phases, dtype=torch.float64):
        """
        Constructor
        :param sample_len: Length of the time-series in time steps.
        :param n_samples: Number of samples to generate.
        :param a:
        :param b:
        :param c:
        """
        # Properties
        self.sample_len = sample_len
        self.n_samples = n_samples
        self.dtype = dtype
        self.sine1 = lambda n: amplitudes[0] * math.sin(2.0 * math.pi * (n + phases[0]) / periods[0])
        self.sine2 = lambda n: amplitudes[1] * math.sin(2.0 * math.pi * (n + phases[1]) / periods[1])

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
    # PUBLIC
    ##############################################

    # Regenerate
    def regenerate(self):
        """
        Regenerate
        :return:
        """
        # Generate data set
        self.outputs = self._generate()
    # end regenerate

    ##############################################
    # PRIVATE
    ##############################################

    # Random initial points
    def random_initial_points(self):
        """
        Random initial points
        :return:
        """
        # Set
        return np.random.random() * (math.pi * 2.0)
    # end random_initial_points

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
            sample = torch.zeros(self.sample_len, 1, dtype=self.dtype)

            # Time steps
            for t in range(self.sample_len):
                sample[t, 0] = self.sine1(i * self.sample_len + t) + self.sine2(i * self.sample_len + t)
            # end for

            # Append
            samples.append(sample)
        # end for

        return samples
    # end _generate

# end MixedSinesDataset
