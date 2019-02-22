# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset
from random import shuffle
import numpy as np


# Henon Attractor
class HenonAttractor(Dataset):
    """
    The Rössler attractor is the attractor for the Rössler system, a system of three non-linear ordinary differential
    equations originally studied by Otto Rössler. These differential equations define a continuous-time dynamical
    system that exhibits chaotic dynamics associated with the fractal properties of the attractor.
    """

    # Constructor
    def __init__(self, sample_len, n_samples, xy, a, b, washout=0, normalize=False, seed=None):
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
        self.a = a
        self.b = b
        self.xy = xy
        self.normalize = normalize
        self.washout = washout

        # Seed
        if seed is not None:
            torch.initial_seed(seed)
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

    # Henon
    def _henon(self, x, y):
        """
        Henon
        :param x:
        :param y:
        :param z:
        :return:
        """
        x_dot = 1 - (self.a * (x * x)) + y
        y_dot = self.b * x
        return x_dot, y_dot
    # end _lorenz

    # Generate
    def _generate(self):
        """
        Generate dataset
        :return:
        """
        # Sizes
        total_size = self.sample_len

        # First position
        xy = self.xy

        # Samples
        samples = list()

        # Washout
        for t in range(self.washout):
            xy = self._henon(xy[0], xy[1])
        # end for

        # For each sample
        for n in range(self.n_samples):
            # Tensor
            sample = torch.zeros(total_size, 2)

            # Timesteps
            for t in range(total_size):
                xy = self._henon(xy[0], xy[1])
                sample[t] = xy
            # end for

            # Normalize
            if self.normalize:
                maxval = torch.max(sample, dim=0)
                minval = torch.min(sample, dim=0)
                sample = torch.mm(torch.inv(torch.diag(maxval - minval)), (sample - minval.repeat(total_size, 1)))
            # end if

            # Add
            samples.append(sample)
        # end for

        # Shuffle
        shuffle(samples)

        return samples
    # end _generate

# end HenonAttractor
