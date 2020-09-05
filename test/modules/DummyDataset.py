# -*- coding: utf-8 -*-
#
# File : test/modules/DummyDataset.py
# Description : Dummy dataset with random data and a specific number of classes.
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


# Dummy dataset with random data and a specific number of classes.
class DummyDataset(Dataset):
    """
    Dummy dataset with random data and a specific number of classes.
    """

    # Constructor
    def __init__(self, sample_len, n_samples, input_dim, n_classes, dtype=torch.float64):
        """
        Constructor
        :param sample_len: Timeseries lengths.
        :param n_samples: Number of samples to generate.
        :param input_dim: Timeseries dimensions.
        :param dtype: Data type.
        """
        # Properties
        self.sample_len = sample_len
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.dtype = dtype
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
        :return: Sample
        """
        # Class
        sample_class = idx % self.n_classes

        # Return sample
        return torch.randn(size=(self.sample_len, self.input_dim), dtype=self.dtype), torch.LongTensor([sample_class])
    # end __getitem__

    #endregion OVERRIDE

# end DummyDataset
