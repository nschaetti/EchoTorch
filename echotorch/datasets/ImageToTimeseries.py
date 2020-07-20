# -*- coding: utf-8 -*-
#
# File : echotorch/datasets/ImageToTimeseries.py
# Description : Transform a dataset of images into timeseries.
# Date : 6th of November, 2019
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
from torch.utils.data import Dataset


# Image to timeseries dataset
class ImageToTimeseries(Dataset):
    """
    Image to timeseries dataset
    """

    # Constructor
    def __init__(self, image_dataset, n_images, transpose=True):
        """
        Constructor
        :param image_dataset: A Dataset object to transform
        :param n_images: How many images to join to compose a timeserie ?
        :param time_axis: Time dimension
        :param transpose: Transpose image before concatenating (start from side if true, from top if false)
        """
        # Params
        self._image_dataset = image_dataset
        self._n_images = n_images
        self._tranpose = transpose
    # end __init__

    #region OVERRIDE

    # To string
    def __str__(self):
        """
        To string
        :return: String version of the object
        """
        str_object = "Dataset ImageToTimeseries\n"
        str_object += "\tN. images : {}\n".format(self._n_images)
        str_object += "\tDataset : {}".format(str(self._image_dataset))
        return str_object
    # end __str__

    # Length
    def __len__(self):
        """
        Length
        :return: How many samples
        """
        return int(math.ceil(len(self._image_dataset) / self._n_images))
    # end __len__

    # Get item
    def __getitem__(self, item):
        """
        Get item
        :param item: Item index
        :return: (sample, target)
        """
        # Data and target
        timeseries_data = None
        timeseries_target = None

        # Get samples
        for i in range(self._n_images):
            # Get sample
            sample_data, sample_target = self._image_dataset[item * self._n_images + i]

            # Transpose
            if self._tranpose:
                sample_data = sample_data.permute(0, 2, 1)
            # end if

            # To tensor if numeric
            if isinstance(sample_target, int) or isinstance(sample_target, float):
                sample_target = torch.LongTensor([[sample_target]])
            elif isinstance(sample_target, torch.Tensor):
                sample_target = sample_target.reshape((1, -1))
            # end if

            # Concat
            if i == 0:
                timeseries_data = sample_data
                timeseries_target = sample_target
            else:
                timeseries_data = torch.cat((timeseries_data, sample_data), axis=1)
                timeseries_target = torch.cat((timeseries_target, sample_target), axis=0)
            # end for
        # end for

        return timeseries_data, timeseries_target
    # end __getitem__

    #endregion OVERRIDE

# end ImageToTimeseries
