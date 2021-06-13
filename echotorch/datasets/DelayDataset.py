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
from typing import Union, List, Tuple

# Local imports
from .EchoDataset import EchoDataset


# Create a version of a dataset with delayed inputs.
class DelayDataset(EchoDataset):
    """
    Generates a series of input time series and delayed versions as outputs.
    Delay is given in number of time steps. Can be used to empirically measure the
    memory capacity of a system.
    """

    # region CONSTUCTORS

    # Constructor
    def __init__(self, root_dataset, n_delays=10, data_index=0, keep_indices=None):
        """
        Constructor
        :param root_dataset: Root dataset
        :param n_delays: Number of step to delay
        """
        # Properties
        self._root_dataset = root_dataset
        self._n_delays = n_delays
        self._data_index = data_index
        self._keep_indices = keep_indices
    # end __init__

    # endregion CONSTRUCTORS

    # region OVERRIDE

    # Get the whole dataset
    @property
    def data(self) -> Union[Tuple[List, List], Tuple[List, List, List]]:
        """
        Get the whole dataset (according to init parameters)
        @return: The Torch Tensor
        """
        # List of samples
        samples_in = list()
        samples_out = list()
        samples_index = list()

        # For each sample in the dataset
        for idx in range(len(self._root_dataset)):
            sample = self[idx]
            if self._keep_indices:
                samples_in.append(sample[0])
                samples_out.append(sample[1])
                samples_index.append(sample[2])
            else:
                samples_in.append(sample[0])
                samples_out.append(sample[1])
            # end if
        # end for

        # Return
        if self._keep_indices:
            return samples_in, samples_out, samples_index
        else:
            return samples_in, samples_out, samples_index
        # end if
    # end data

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
        @param idx: Item index
        @return:
        """
        # Item list
        item_list = list()

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
            output_timeseries = original_timeseries.clone()
        # end if

        # Add input and output
        item_list.append(input_timeseries)
        item_list.append(output_timeseries)

        # Add all additional data
        if self._keep_indices is not None:
            for keep_index in self._keep_indices:
                item_list.append(data[keep_index])
            # end for
        # end if

        return item_list
    # end __getitem__

    # Extra representation
    def extra_repr(self) -> str:
        """
        Extra representation
        """
        return "root_dataset={}, n_delays={}, data_index={}, keep_indices={}".format(
            self._root_dataset,
            self._n_delays,
            self._data_index,
            self._keep_indices
        )
    # end extra_repr

    # Function to generate data
    def datafunc(self, idx) -> List:
        """
        Function to generate data
        @param idx: Item index
        @return: Timeseries, delay timeseries
        """
        # Item list
        item_list = list()

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

        # Add input and output
        item_list.append(input_timeseries)
        item_list.append(output_timeseries)

        # Add all additional data
        if self._keep_indices is not None:
            for keep_index in self._keep_indices:
                item_list.append(data[keep_index])
            # end for
        # end if

        return item_list
    # end datafunc

    # endregion OVERRIDE

# end DelayDataset
