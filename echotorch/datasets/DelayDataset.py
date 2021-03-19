# -*- coding: utf-8 -*-
#
# File : echotorch/datasets/DelayDataset.py
# Description : Create a version of a dataset with delayed inputs.
# Date : 17th of Marche, 2021
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
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>,
# University of Geneva <nils.schaetti@unige.ch>

# Imports
import torch
from torch.utils.data.dataset import Dataset


# Create a version of a dataset with delayed inputs.
class DelayDataset(Dataset):
    """
    Generates a series of input timeseries and delayed versions as outputs.
    Delay is given in number of timesteps. Can be used to empirically measure the
    memory capacity of a system.
    """

    # region CONSTUCTORS

    # Constructor
    def __init__(self, root_dataset, n_delays=10, data_index=0):
        """
        Constructor
        :param root_dataset: Root dataset
        :param n_delays: Number of step to delay
        """
        # Properties
        self._root_dataset = root_dataset
        self._n_delays = n_delays
        self._data_index = data_index
    # end __init__

    # endregion CONSTRUCTORS

    # region OVERRIDE

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return len(self._root_dataset)
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        # Get sample from root dataset
        data = self._root_dataset[idx]

        # Get data
        original_timeseries = data[self._data_index]

        # Future
        if self._n_delays > 0:
            # Prediction
            input_timeseries = original_timeseries[:-self._n_delays]
            output_timeseries = original_timeseries[self._n_delays:]
        elif self._n_delays < 0:
            # Memory
            input_timeseries = original_timeseries[self._n_delays:]
            output_timeseries = original_timeseries[:-self._n_delays]
        else:
            input_timeseries = original_timeseries
            output_timeseries = original_timeseries.copy()
        # end if

        return input_timeseries, output_timeseries
    # end __getitem__

    # endregion OVERRIDE

# end DelayDataset
