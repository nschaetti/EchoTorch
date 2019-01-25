# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset
import numpy as np


# Lorenz Attractor
class LorenzAttractor(Dataset):
    """
    The Rössler attractor is the attractor for the Rössler system, a system of three non-linear ordinary differential
    equations originally studied by Otto Rössler. These differential equations define a continuous-time dynamical
    system that exhibits chaotic dynamics associated with the fractal properties of the attractor.
    """

    # Constructor
    def __init__(self, sample_len, n_samples, s=10.0, r=28.0, b=2.667, init_mult=5.0, dt=0.01, normalize=False, mult=1.0, dim=3, seed=None):
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
        self.s = s
        self.r = r
        self.b = b
        self.dt = dt
        self.normalize = normalize
        self.mult = mult
        self.init_mult = init_mult
        self.mu = {1: 0.0507, 2: 0.3929, 3: 8.7584}
        self.std = {1: 8.4643, 2: 9.0728, 3: 14.2691}
        self.dim = dim

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

    # Lorenz
    def _lorenz(self, x, y, z):
        """
        Lorenz
        :param x:
        :param y:
        :param z:
        :return:
        """
        x_dot = self.s * (y - x)
        y_dot = self.r * x - y - x * z
        z_dot = x * y - self.b * z
        return x_dot, y_dot, z_dot
    # end _lorenz

    # Random initial points
    def random_initial_points(self):
        """
        Random initial points
        :return:
        """
        # Set
        return (np.random.random() * 2.0 - 1.0) * self.init_mult, \
               (np.random.random() * 2.0 - 1.0) * self.init_mult, \
               (np.random.random() * 2.0 - 1.0) * self.init_mult
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
            sample = torch.zeros(self.sample_len, 3)

            # Set
            init_x, init_y, init_z = self.random_initial_points()
            sample[0, 0] = init_x
            sample[0, 1] = init_y
            sample[0, 2] = init_z

            for t in range(1, self.sample_len):
                # Derivatives of the X, Y, Z state
                x_dot, y_dot, z_dot = self._lorenz(sample[t - 1, 0], sample[t - 1, 1], sample[t - 1, 2])

                # Set
                sample[t, 0] = sample[t - 1, 0] + (x_dot * self.dt)
                sample[t, 1] = sample[t - 1, 1] + (y_dot * self.dt)
                sample[t, 2] = sample[t - 1, 2] + (z_dot * self.dt)
            # end for

            # Keep only specific dim
            sample = sample[:, :self.dim]

            # Normalize
            if self.normalize:
                sample = ((sample - self.mu[self.dim]) / self.std[self.dim]) * self.mult
            # end if

            # Append
            samples.append(sample)
        # end for

        return samples
    # end _generate

# end LorenzAttractor
