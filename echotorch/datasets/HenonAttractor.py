# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset
from random import shuffle


# Henon Attractor
class HenonAttractor(Dataset):
    """
    The Rössler attractor is the attractor for the Rössler system, a system of three non-linear ordinary differential
    equations originally studied by Otto Rössler. These differential equations define a continuous-time dynamical
    system that exhibits chaotic dynamics associated with the fractal properties of the attractor.
    """

    # Constructor
    def __init__(self, sample_len, n_samples, a=1.4, b=0.3, x=0.0, y=0.0, start=0, dt=0.01, normalize=False, mult=1.0):
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
        self.init_x = x
        self.init_y = y
        self.dt = dt
        self.normalize = normalize
        self.mult = mult
        self.start = start

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
        # List of samples
        outputs = list()

        # Sizes
        total_size = self.start + self.sample_len * self.n_samples

        # Tensor
        samples = torch.zeros(total_size, 2)

        # First position
        samples[0, 0] = self.init_x
        samples[0, 1] = self.init_y

        for t in range(1, total_size):
            # Derivatives of the X, Y, Z state
            x_dot, y_dot = self._henon(samples[t - 1, 0], samples[t - 1, 1])
            samples[t, 0] = samples[t - 1, 0] + (x_dot * self.dt)
            samples[t, 1] = samples[t - 1, 1] + (y_dot * self.dt)
        # end for

        # Normalize
        if self.normalize:
            mu = torch.mean(samples)
            std = torch.std(samples)
            samples = ((samples - mu) / std) * self.mult
        # end if

        # For each samples
        for i in range(self.start, total_size, self.sample_len):
            outputs.append(samples[i:i + self.sample_len])
        # end for

        # Shuffle
        shuffle(outputs)

        return outputs
    # end _generate

# end HenonAttractor
