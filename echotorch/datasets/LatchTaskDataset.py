# -*- coding: utf-8 -*-
#
# File : echotorch/datasets/LatchTaskDataset.py
# Description : Dataset for the latch task
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


# Latch task dataset
class LatchTaskDataset(Dataset):
    """
    Latch task dataset
    """

    # Constructor
    def __init__(self, n_samples, length_min, length_max, n_pics, dtype=torch.float32):
        """
        Constructor
        :param sample_len: Sample's length
        :param period:
        """
        # Properties
        self.length_min = length_min
        self.length_max = length_max
        self.n_samples = n_samples
        self.n_pics = n_pics
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
            sample_inputs = torch.zeros((1, sample_len), dtype=self.dtype)
            sample_outputs = torch.zeros((1, sample_len), dtype=self.dtype)

            # List of pic position
            pic_positions = list()

            # Generate random positions
            for p in range(self.n_pics):
                # Random pic position
                random_pic_position = torch.randint(high=sample_len, size=(1,)).item()

                # Save pic position
                pic_positions.append(random_pic_position)

                # Set pic in input
                sample_inputs[0, random_pic_position] = 1.0
            # end for

            # Order pic positions
            pic_positions.sort()

            # For each pic
            first_pic = True
            for i in range(self.n_pics):
                if first_pic:
                    if i == self.n_pics - 1:
                        pic_pos1 = pic_positions[i]
                        pic_pos2 = sample_len
                    else:
                        pic_pos1 = pic_positions[i]
                        pic_pos2 = pic_positions[i+1] + 1
                    # end if

                    # Length of segment
                    segment_length = pic_pos2 - pic_pos1

                    # Set in outputs
                    sample_outputs[0, pic_pos1:pic_pos2] = torch.ones(segment_length, dtype=self.dtype)

                    # Not first pic
                    first_pic = False
                else:
                    first_pic = True
                # end if
            # end for

            # Append
            samples.append((sample_inputs, sample_outputs))
        # end for

        return samples
    # end _generate

    #endregion PRIVATE

# end LatchTaskDataset
