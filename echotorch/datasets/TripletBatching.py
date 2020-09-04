# -*- coding: utf-8 -*-
#
# File : datasets/TripletBatching.py
# Description : Take a dataset with different classes and create a dataset of triplets with an anchor (A) and positive
# example (same class) and a negative one (different class).
# Date : 21th of July, 2020
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
# Copyright Nils Schaetti <nils.schaetti@unine.ch>, <nils.schaetti@unige.ch>


# Imports
import random
from torch.utils.data import Dataset


# Triplet batching
class TripletBatching(Dataset):
    """
    Take a dataset with different classes and create a dataset of triplets with an anchor (A) and positive
    example (same class) and a negative one (different class).
    """

    # Constructor
    def __init__(self, root_dataset, data_index, target_index, target_count, n_samples,
                 target_type='int', *args, **kwargs):
        """
        Constructor
        :param root_dataset: The main dataset
        """
        # Call upper class
        super(TripletBatching, self).__init__(*args, **kwargs)

        # Properties
        self._root_dataset = root_dataset
        self._data_index = data_index
        self._target_index = target_index
        self._target_count = target_count
        self._n_samples = n_samples
        self._target_type = target_type

        # Item indices for each target classes
        self._targets_indices = {}
        self._targets_indices_len = {}
        for t_i in range(self._target_count):
            self._targets_indices[t_i] = list()
            self._targets_indices_len[t_i] = 0
        # end for

        # List of target classes
        self._target_classes = list()

        # Analyse the dataset
        self._analyse_dataset()
    # end __init__

    #region PRIVATE

    # Analyze the root dataset to determine the total number of samples
    def _analyse_dataset(self):
        """
        Analyse the root dataset
        """
        # For each samples
        for data_i in range(len(self._root_dataset)):
            # Data
            data = self._root_dataset[data_i]

            # Get target class
            target_class = data[self._target_index]

            # Transform to key
            if self._target_type == 'tensor':
                target_class = target_class.item()
            # end if

            # Save index
            self._targets_indices[target_class].append(data_i)
            self._targets_indices_len[target_class] += 1

            # Add to list of target classes
            # As we have at least on example
            if target_class not in self._target_classes:
                self._target_classes.append(target_class)
            # end if
        # end for
    # end _analyse_dataset

    #endregion PRIVATE

    #region OVERRIDE

    # Length of the dataset
    def __len__(self):
        """
        Length of the dataset
        :return: How many samples
        """
        return self._n_samples
    # end __len__

    # Get a sample in the dataset
    def __getitem__(self, item):
        """
        Get a sample in the dataset
        :param item: Item index (start 0)
        :return: Dataset sample
        """
        # Number of classes
        classes_count = len(self._target_classes)

        # Choose a random anchor class
        anchor_class = self._target_classes[random.randrange(classes_count)]

        # Indices of anchor class
        anchor_class_indices = self._targets_indices[anchor_class]
        anchor_class_indices_count = len(anchor_class_indices)

        # Choose a random anchor
        anchor_index = anchor_class_indices[random.randrange(anchor_class_indices_count)]
        anchor_sample = self._root_dataset[anchor_index]

        # Choose a random positive example
        anchor_class_indices.remove(anchor_index)
        positive_index = anchor_class_indices[random.randrange(anchor_class_indices_count-1)]
        positive_sample = self._root_dataset[positive_index]

        # Choose a random negative class
        targets_classes = self._target_classes.copy()
        targets_classes.remove(anchor_class)
        negative_class = targets_classes[random.randrange(classes_count-1)]

        # Indices of negative class
        negative_class_indices = self._targets_indices[negative_class]
        negative_class_indices_count = len(negative_class_indices)

        # Choose a random negative example
        negative_index = negative_class_indices[random.randrange(negative_class_indices_count)]
        negative_sample = self._root_dataset[negative_index]

        return anchor_sample, positive_sample, negative_sample
    # end __getitem__

    #endregion OVERRIDE

# end TripletBatching
