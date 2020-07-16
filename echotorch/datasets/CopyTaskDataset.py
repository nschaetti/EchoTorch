# -*- coding: utf-8 -*-
#
# File : echotorch/datasets/CopyTaskDataset.py
# Description : Dataset for the copy task (Graves et al, 2016)
# Date : 16th of July, 2020
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
import numpy as np


# Copy task dataset
class CopyTaskDataset(Dataset):
    """
    Copy task dataset
    """

    # Constructor
    def __init__(self, n_samples, length_min, length_max, n_inputs, dtype=torch.float32):
        """
        Constructor
        :param sample_len: Sample's length
        :param period:
        """
        # Properties
        self.length_min = length_min
        self.length_max = length_max
        self.n_samples = n_samples
        self.n_inputs = n_inputs
        self.dtype = dtype

        # Generate data set
        self.samples = self._generate()
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
        return self.samples[idx]
    # end __getitem__

    #endregion OVERRIDE

    #region PRIVATE

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
            # Generate length
            sample_len = torch.randint(low=self.length_min, high=self.length_max, size=(1,)).item()

            # Create empty inputs and output
            sample_inputs = torch.zeros((sample_len * 2 + 1, self.n_inputs + 1), dtype=self.dtype)
            sample_outputs = torch.zeros((sample_len * 2 + 1, self.n_inputs + 1), dtype=self.dtype)

            # Generate a random pattern
            random_pattern = torch.randint(low=0, high=2, size=(sample_len, self.n_inputs))

            # Set in inputs and outputs
            sample_inputs[:sample_len, :self.n_inputs] = random_pattern
            sample_outputs[sample_len+1:, :self.n_inputs] = random_pattern
            sample_inputs[sample_len, self.n_inputs] = 1.0

            # Append
            samples.append((sample_inputs, sample_outputs))
        # end for

        return samples
    # end _generate

    #endregion PRIVATE

# end CopyTaskDataset
