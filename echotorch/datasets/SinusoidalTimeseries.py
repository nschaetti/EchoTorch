# -*- coding: utf-8 -*-
#
# File : echotorch/datasets/SinusoidalTimeseries.py
# Description : Create a dataset from sine patterns.
# Date : 5th of September, 2020
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
from torch.utils.data.dataset import Dataset
import math
import numpy as np


# Sinusoidal Timeseries
class SinusoidalTimeseries(Dataset):
    """
    Sinusoidal timeseries
    """

    # Constructor
    def __init__(self, sample_len, n_samples, period, a=1.0, m=0.0, start=1, dtype=torch.float64):
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
        self.func = lambda n: a * math.sin(2.0 * math.pi * (n + start) / period) + m

        # Generate data set
        self.outputs = self._generate()
    # end __init__

    #region OVERRIDE

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

    #endregion OVERRIDE

    #region PUBLIC

    # Regenerate
    def regenerate(self):
        """
        Regenerate
        :return:
        """
        # Generate data set
        self.outputs = self._generate()
    # end regenerate

    #endregion PUBLIC

    #region PRIVATE

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
                sample[t, 0] = self.func(i * self.sample_len + t)
            # end for

            # Append
            samples.append(sample)
        # end for

        return samples
    # end _generate

    #endregion PRIVATE

# end SinusoidalTimeseries
