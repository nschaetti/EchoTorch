# -*- coding: utf-8 -*-
#
# File : echotorch/datasets/MemTestDataset.py
# Description : Base class for EchoTorch datasets
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

# Local imports
from .EchoDataset import EchoDataset


# Generates a series of input timeseries and delayed versions as outputs.
class MemTestDataset(EchoDataset):
    """
    Generates a series of input time series and delayed versions as outputs.
    Delay is given in number of time steps. Can be used to empirically measure the
    memory capacity of a system.
    """

    # Constructor
    def __init__(self, sample_len, n_samples, n_delays=10, seed=None):
        """
        Constructor
        :param sample_len: Length of the time-series in time steps.
        :param n_samples: Number of samples to generate.
        :param n_delays: Number of step to delay
        :param seed: Seed of random number generator.
        """
        # Properties
        self.sample_len = sample_len
        self.n_samples = n_samples
        self.n_delays = n_delays

        # Init seed if needed
        if seed is not None:
            torch.manual_seed(seed)
        # end if
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
        inputs = (torch.rand(self.sample_len, 1) - 0.5) * 1.6
        outputs = torch.zeros(self.sample_len, self.n_delays)
        for k in range(self.n_delays):
            outputs[:, k:k+1] = torch.cat((torch.zeros(k + 1, 1), inputs[:-k - 1, :]), dim=0)
        # end for
        return inputs, outputs
    # end __getitem__

    # endregion OVERRIDE

# end MemTestDataset
