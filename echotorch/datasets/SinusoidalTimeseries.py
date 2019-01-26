# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset
import math
from random import shuffle
import numpy as np


# Sinusoidal Timeseries
class SinusoidalTimeseries(Dataset):
    """
    The Rössler attractor is the attractor for the Rössler system, a system of three non-linear ordinary differential
    equations originally studied by Otto Rössler. These differential equations define a continuous-time dynamical
    system that exhibits chaotic dynamics associated with the fractal properties of the attractor.
    """

    # Constructor
    def __init__(self, sample_len, n_samples, w, a=1, g=None, seed=None):
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
        self.w = w
        self.a = a
        self.g = g

        # Seed
        if seed is not None:
            np.random.seed(seed)
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
            sample = torch.zeros(self.sample_len, 1)

            # Random start
            if self.g is None:
                # Init
                init_g = self.random_initial_points()

                for t in range(0, self.sample_len):
                    sample[t, 0] = self.a * math.sin(self.w * t + init_g)
                # end for
            else:
                for t in range(0, self.sample_len):
                    sample[t, 0] = self.a * math.sin(self.w * t + self.g)
                # end for
            # end if

            # Append
            samples.append(sample)
        # end for

        return samples
    # end _generate

# end SinusoidalTimeseries
