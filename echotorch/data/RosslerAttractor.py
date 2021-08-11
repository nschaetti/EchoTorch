# -*- coding: utf-8 -*-
#
# File : echotorch/datasets/RosslerAttractor.py
# Description : Rossler Attractor
# Date : 25th of January, 2021
#
# This file is part of EchoTorch.  EchoTorch is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Nils Schaetti <nils.schaetti@unine.ch>

# Imports
import torch
import numpy as np

# Local imports
from .EchoDataset import EchoDataset


# Rossler Attractor
class RosslerAttractor(EchoDataset):
    """
    The Rössler attractor is the attractor for the Rössler system, a system of three non-linear ordinary differential
    equations originally studied by Otto Rössler. These differential equations define a continuous-time dynamical
    system that exhibits chaotic dynamics associated with the fractal properties of the attractor.
    """

    # Constructor
    def __init__(self, sample_len, n_samples, xyz, a, b, c, dt=0.01, washout=0, normalize=False, seed=None):
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
        self.c = c
        self.dt = dt
        self.normalize = normalize
        self.washout = washout
        self.xyz = xyz

        # Seed
        if seed is not None:
            np.random.seed(seed)
        # end if

        # Generate data set
        self.outputs = self._generate()
    # end __init__

    # region OVERRIDE

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

    # endregion OVERRIDE

    # region PUBLIC

    # Regenerate
    def regenerate(self):
        """
        Regenerate
        :return:
        """
        # Generate data set
        self.outputs = self._generate()
    # end regenerate

    # endregion PUBLIC

    # region PRIVATE

    # Rossler
    def _rossler(self, x, y, z):
        """
        Lorenz
        :param x:
        :param y:
        :param z:
        :return:
        """
        x_dot = -(y + z)
        y_dot = x + self.a * y
        z_dot = self.b + x * z - self.c * z
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
            x_dot, y_dot, z_dot = self._rossler(xyz[0], xyz[1], xyz[2])

            # Apply changes
            xyz[0] += self.dt * x_dot
            xyz[1] += self.dt * y_dot
            xyz[2] += self.dt * z_dot
        # end for

        # For each sample
        for i in range(self.n_samples):
            # Tensor
            sample = torch.zeros(total_size, 3)

            # Time steps
            for t in range(1, self.sample_len):
                # Derivatives of the X, Y, Z state
                x_dot, y_dot, z_dot = self._rossler(xyz[0], xyz[1], xyz[2])

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

    # endregion PRIVATE

# end RosslerAttractor
