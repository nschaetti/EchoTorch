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
    def __init__(self, sample_len, n_samples, xyz, sigma, b, r, dt=0.01, washout=0, normalize=False, seed=None):
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
        self.xyz = xyz
        self.dt = dt
        self.normalize = normalize
        self.washout = washout
        self.sigma = sigma
        self.b = b
        self.r = r

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

    # Lorenz
    def _lorenz(self, x, y, z):
        """
        Lorenz
        :param x:
        :param y:
        :param z:
        :return:
        """
        x_dot = self.sigma * (y - x)
        y_dot = self.r * x - y - x * z
        z_dot = x * y - self.b * z
        return x_dot, y_dot, z_dot
    # end _lorenz

    # Generate
    def _generate(self):
        """
        Generate dataset
        :return:
        """
        # Sizes
        total_size = self.sample_len

        # List of samples
        samples = list()

        # XYZ
        xyz = self.xyz

        # Washout
        for t in range(self.washout):
            # Derivatives of the X, Y, Z state
            x_dot, y_dot, z_dot = self._lorenz(xyz[0], xyz[1], xyz[2])

            # Apply changes
            xyz[0] += self.dt * x_dot
            xyz[1] += self.dt * y_dot
            xyz[2] += self.dt * z_dot
        # end for

        # For each sample
        for i in range(self.n_samples):
            # Tensor
            sample = torch.zeros(self.sample_len, 3)
            for t in range(self.sample_len):
                # Derivatives of the X, Y, Z state
                x_dot, y_dot, z_dot = self._lorenz(xyz[0], xyz[1], xyz[2])

                # Apply changes
                xyz[0] += self.dt * x_dot
                xyz[1] += self.dt * y_dot
                xyz[2] += self.dt * z_dot

                # Set
                sample[t, 0] = xyz[0]
                sample[t, 1] = xyz[1]
                sample[t, 2] = xyz[2]
            # end for

            # Normalize
            if self.normalize:
                maxval = torch.max(sample, dim=0)
                minval = torch.min(sample, dim=0)
                sample = torch.mm(torch.inv(torch.diag(maxval - minval)), (sample - minval.repeat(total_size, 1)))
            # end if

            # Append
            samples.append(sample)
        # end for

        return samples
    # end _generate

# end LorenzAttractor
