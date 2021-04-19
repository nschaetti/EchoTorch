# -*- coding: utf-8 -*-
#
# File : echotorch/datasets/CopyTaskDataset.py
# Description : Dataset for the copy task (Graves et al, 2016)
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
from typing import List, Tuple

# Imports local
from .EchoDataset import EchoDataset


# Copy task dataset
class CopyTaskDataset(EchoDataset):
    """
    Copy task dataset
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, n_samples, length_min, length_max, n_inputs, dtype=None):
        """
        Constructor
        @param n_samples: How many samples to generate
        @param length_min: Minimum length of the series
        @param length_max: Maximum length of the series
        @param n_inputs: How many inputs
        @param dtype: Data type
        """
        # Properties
        self.length_min = length_min
        self.length_max = length_max
        self.n_samples = n_samples
        self.n_inputs = n_inputs
        self.dtype = dtype

        # Generate data set
        self.samples = self.generate(
            self.n_samples,
            self.length_min,
            self.length_max,
            self.n_inputs,
            self.dtype
        )
    # end __init__

    # endregion CONSTRUCTORS

    # region PUBLIC

    # Get the whole dataset
    @property
    def data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Get the whole dataset as a list
        @return: A tuple of list
        """
        return self.samples
    # end data

    # endregion PUBLIC

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
        return self.samples[idx]
    # end __getitem__

    # Extra representation
    def extra_repr(self) -> str:
        """
        Extra representation
        @return: Dataset representation as a string
        """
        return "length_min={}, length_max={}, n_inputs={}, dtype={}".format(
            self.length_min,
            self.length_max,
            self.n_inputs,
            self.dtype
        )
    # end extra_repr

    # Function to generate a sample
    def datafunc(self, length_min, length_max, n_inputs, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function to generate a sample
        @param length_min: Minimum length
        @param length_max:
        @param n_inputs:
        @param dtype: Data type
        @return:
        """
        # Generate length
        sample_len = torch.randint(low=length_min, high=length_max, size=(1,)).item()

        # Create empty inputs and output
        sample_inputs = torch.zeros((sample_len * 2 + 1, n_inputs + 1), dtype=dtype)
        sample_outputs = torch.zeros((sample_len * 2 + 1, n_inputs + 1), dtype=dtype)

        # Generate a random pattern
        random_pattern = torch.randint(low=0, high=2, size=(sample_len, n_inputs))

        # Set in inputs and outputs
        sample_inputs[:sample_len, :n_inputs] = random_pattern
        sample_outputs[sample_len + 1:, :n_inputs] = random_pattern
        sample_inputs[sample_len, n_inputs] = 1.0

        return sample_inputs, sample_outputs
    # end datafunc

    # endregion OVERRIDE

    # region STATIC

    # Generate samples
    def generate(
            self,
            n_samples,
            length_min,
            length_max,
            n_inputs,
            dtype=None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate samples
        :param n_samples:
        :param length_min:
        :param length_max:
        :param n_inputs:
        :param dtype:
        """
        # List of samples
        samples_in = list()
        samples_out = list()

        # For each sample
        for i in range(n_samples):
            # Generate a sample
            sample_inputs, sample_outputs = self.datafunc(length_min, length_max, n_inputs, dtype)

            # Append
            samples_in.append(sample_inputs)
            samples_out.append(sample_outputs)
        # end for

        return samples_in, samples_out
    # end generate

    # endregion STATIC

# end CopyTaskDataset
