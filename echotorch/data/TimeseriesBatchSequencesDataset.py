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
import math
import torch

# Local imports
from .EchoDataset import EchoDataset


# Timeseries batch cutting
class TimeseriesBatchSequencesDataset(EchoDataset):
    """
    Take a dataset of timeseries and cut all of them by window size and compose batches
    """

    # Constructor
    def __init__(self, root_dataset, window_size, data_indices, stride, remove_indices, time_axis=0,
                 dataset_in_memory=False, *args, **kwargs):
        """
        Constructor
        :param root_dataset: Root dataset
        :param window_size: Sequence size in the timeseries
        :param data_indices: Which output of dataset is a timeseries tensor
        :param stride: Stride
        :param time_axis: Which axis is the temporal dimension in the output tensor
        """
        # Call upper class
        super(TimeseriesBatchSequencesDataset, self).__init__(*args, **kwargs)

        # Parameters
        self.root_dataset = root_dataset
        self.window_size = window_size
        self.data_indices = data_indices
        self.stride = stride
        self.time_axis = time_axis
        self.dataset_in_memory = dataset_in_memory
        self.remove_indices = remove_indices

        # Dataset information
        self.timeseries_lengths = list()
        self.timeseries_total_length = 0
        self.root_dataset_n_samples = 0
        self.timeseries_sequences_info = list()
        self.n_samples = 0
        self.dataset_samples = list()

        # Load dataset
        self._load_dataset()
    # end __init__

    # region PRIVATE

    # Load dataset
    def _load_dataset(self):
        """
        Load dataset
        :return:
        """
        # Item position
        item_position = 0

        # Load root dataset
        for item_i in range(len(self.root_dataset)):
            # Get data
            data = self.root_dataset[item_i]

            # Get the first timeserie returned by the dataset
            timeserie_data = data[self.data_indices[0]] if self.data_indices is not None else data

            # Length of timeseries in number of samples (sequences)
            timeserie_length = timeserie_data.size(self.time_axis)

            # timeserie_seq_length = int(math.floor(timeserie_length / self.window_size))
            timeserie_seq_length = int(math.floor((timeserie_length - self.window_size) / self.stride) + 1)

            # Save length and total length
            self.timeseries_lengths.append(timeserie_length)
            self.timeseries_total_length += timeserie_length
            self.timeseries_sequences_info.append({'start': item_position, 'end': item_position + timeserie_seq_length})

            # Keep in memory if asked for
            if self.dataset_in_memory:
                self.dataset_samples.append(data)
            # end if

            # Increment item position
            item_position += timeserie_seq_length
        # end for

        # Total number of samples
        self.n_samples = item_position
    # end _load_dataset

    # endregion PRIVATE

    # region OVERRIDE

    # Get a sample in the dataset
    def __getitem__(self, item):
        """
        Get a sample in the dataset
        :param item: Item index (start 0)
        :return: Dataset sample
        """
        # print("__getitem__ {}".format(item))
        # Go through each samples in the root dataset
        for item_i in range(len(self.root_dataset)):
            # Timeserie info
            ts_start_end = self.timeseries_sequences_info[item_i]

            # The item is in this sample
            if ts_start_end['start'] <= item < ts_start_end['end']:
                # Get the corresponding timeseries
                if self.dataset_in_memory:
                    data = list(self.dataset_samples[item_i]) if self.data_indices is not None else self.dataset_samples[item_i]
                else:
                    data = list(self.root_dataset[item_i]) if self.data_indices is not None else self.root_dataset[item_i]
                # end if

                # Sequence start and end
                # sequence_start = (item - ts_start_end['start']) * self.window_size
                sequence_start = (item - ts_start_end['start']) * self.stride
                sequence_end = sequence_start + self.window_size
                sequence_range = range(sequence_start, sequence_end)

                # For each data to transform
                if self.data_indices is not None:
                    for data_i in self.data_indices:
                        # Get timeserie
                        timeserie_data = data[data_i]

                        # Get sequence according to time axis
                        data[data_i] = torch.index_select(timeserie_data, self.time_axis, torch.tensor(sequence_range))
                    # end for
                else:
                    # Get sequence according to time axis
                    data = torch.index_select(data, self.time_axis, torch.tensor(sequence_range))
                # end if

                # For each data to add batch to
                new_data = list()
                if self.remove_indices is not None:
                    for data_i in range(len(data)):
                        if data_i not in self.remove_indices:
                            new_data.append(data[data_i])
                        # end if
                    # end for
                else:
                    new_data = data
                # end if

                # Return modified data
                return new_data
            # end if
        # end for
    # end __getitem__

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
        return self.n_samples
    # end __len__

    # endregion OVERRIDE

# end TimeseriesBatchSequencesDataset

