# -*- coding: utf-8 -*-
#
# File : datasets/TimeseriesBatchCutting.py
# Description : Take a dataset of timeseries and cut all of them by window size and compose batches
# Date : 20th of July, 2020
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
from torch.utils.data import Dataset


# Timeseries batch cutting
class TimeseriesBatchSequencesDataset(Dataset):
    """
    Take a dataset of timeseries and cut all of them by window size and compose batches
    """

    # Constructor
    def __init__(self, root_dataset, window_size, timeserie_pos=0, time_axis=1, main_axis='sequences'):
        """
        Constructor
        :param root_dataset: Root dataset
        :param window_size: Sequence size in the timeseries
        :param timeserie_pos: Which output of dataset is the timeseries tensor
        :param time_axis: Which axis is the temporal dimension in the output tensor
        :param main_axis: Neighbour sequences come from different samples ('sequences') or from the same (any other value)
        """
        # Parameters
        self.root_dataset = root_dataset
        self.window_size = window_size
        self.timeserie_pos = timeserie_pos
        self.time_axis = time_axis

        # Dataset information
        self.timeseries_lengths = list()
        self.timeseries_total_length = 0
        self.root_dataset_n_samples = 0
        self.n_samples = 0

        # Load dataset
        self._load_dataset()
    # end __init__

    #region PRIVATE

    # Load dataset
    def _load_dataset(self):
        """
        Load dataset
        :return:
        """
        # Load root dataset
        for item_i in range(len(self.root_dataset)):
            # Get data
            data = self.root_dataset[item_i]

            # Get timeserie
            timeserie_data = data[self.timeserie_pos]

            # Save length and total length
            self.timeseries_lengths.append((item_i, timeserie_data.size(self.time_axis)))
            self.timeseries_total_length += timeserie_data.size(self.time_axis)
        # end for
    # end _load_dataset

    #endregion PRIVATE

    #region OVERRIDE

    # To string
    def __str__(self):
        """
        To string
        :return: String version of the object
        """
        str_object = "Dataset TimeseriesBatchSequencesDataset\n"
        str_object += "\tWindow size : {}\n".format(self.window_size)
        return str_object
    # end __str__

    # Length
    def __len__(self):
        """
        Length
        :return: How many samples
        """
        return 0
    # end __len__

    #endregion OVERRIDE

# end TimeseriesBatchSequencesDataset

