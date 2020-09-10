# -*- coding: utf-8 -*-
#
# File : echotorch/datasets/PeriodicSignalDataset.py
# Description : Periodic patterns timeseries
# Date : 10th of September, 2020
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
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>

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
    def __init__(self, sample_len, n_samples, period, start=1, dtype=torch.float64):
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
        self.dtype = dtype

        # Period
        max_val = np.max(period)
        min_val = np.min(period)
        self.rp = 1.8 * (period - min_val) / (max_val - min_val) - 0.9
        self.period_length = len(period)

        # Function
        self.func = lambda n: self.rp[(n + 1) % self.period_length]

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

        # For each sample
        for i in range(self.n_samples):
            # Tensor
            sample = torch.zeros(self.sample_len, 1, dtype=self.dtype)

            # Timestep
            for t in range(self.sample_len):
                sample[t, 0] = self.func(i * self.sample_len + t)
            # end for

            # Append
            samples.append(sample)
        # end for

        return samples
    # end _generate

# end PeriodicSignalDataset
