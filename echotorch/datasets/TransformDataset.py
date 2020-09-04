# -*- coding: utf-8 -*-
#
# File : echotorch/datasets/TransformDataset.py
# Description : Apply a transformation to a dataset.
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
import math
import torch
from torch.utils.data import Dataset


# Transform dataset
class TransformDataset(Dataset):
    """
    Apply a transformation to a dataset.
    """

    # Constructor
    def __init__(self, root_dataset, transform, transform_indices=None, transform_target=None,
                 transform_target_indices=None, *args, **kwargs):
        """
        Constructor
        :param root_dataset: The dataset to transform.
        :param transform: A Transformer object applied to the timeseries.
        :param transform_indices: The indices to select which data returned by the dataset to apply the
        transformation to. Or None if applied directly.
        :param transform_target: A Transformer object applied to the target timeseries.
        :param transform_target_indices: The indices to select which data returned by the dataset to apply the
        target transformation to. Or None if applied directly.
        """
        # Call upper class
        super(TransformDataset, self).__init__(*args, **kwargs)

        # Properties
        self._root_dataset = root_dataset
        self._transform = transform
        self._transform_indices = transform_indices
        self._transform_target = transform_target
        self._transform_target_indices = transform_target_indices
    # end __init__

    #region OVERRIDE

    # Length of the dataset
    def __len__(self):
        """
        Length of the dataset
        :return: Length of the dataset
        """
        return len(self._root_dataset)
    # end __len__

    # Get item
    def __getitem__(self, item):
        """
        Get item
        :param item: Index
        :return:
        """
        # Get data from the item
        item_data = self._root_dataset[item]

        # Transform each inputs
        if self._transform is not None:
            if self._transform_indices is not None:
                for data_i in self._transform_indices:
                    item_data[data_i] = self._transform(item_data[data_i])
                # end for
            else:
                item_data = self._transform(item_data)
            # end if
        # end if

        # Transform each outputs
        if self._transform_target is not None:
            if self._transform_target_indices is not None:
                for data_i in self._transform_target_indices:
                    item_data[data_i] = self._transform_target(item_data[data_i])
                # end for
            else:
                item_data = self._transform_target(item_data)
            # end if
        # end if

        return item_data
    # end __getitem__

    #endregion OVERRIDE

# end TransformDataset
