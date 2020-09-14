# -*- coding: utf-8 -*-
#
# File : echotorch/datasets/RandomSymbolDataset.py
# Description : Create sequence of symbol chosen randomly.
# Date : 10th of September, 2020
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
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>

# Imports
import torch
from torch.utils.data.dataset import Dataset


# Sequences of symbols taken randomly
class RandomSymbolDataset(Dataset):
    """
    Sequences of symbols taken randomly
    """

    # Constructor
    def __init__(self, sample_len, n_samples, vocabulary_size, random_func=None):
        """
        Constructor
        :param sample_len: Length of sequences
        :param n_samples: Number of samples to generate
        :param vocabulary_size: How many symbols in the vocabulary
        :param random_func: Random function to call to choose symbols, if None taken a random generator from pytorch
        :param dtype: Data type (float or double)
        """
        # Properties
        self._sample_len = sample_len
        self._n_samples = n_samples
        self._vocabulary_size = vocabulary_size
        self._random_func = random_func

        # Generate samples
        self._samples = self._generate()
    # end __init__

    # region PRIVATE

    # Generate samples
    def _generate(self):
        """
        Generate samples
        :return: Generated samples
        """
        samples = list()
        for sample_i in range(self._n_samples):
            samples.append(torch.randint(low=0, high=10, size=(self._sample_len, 1)))
        # end for
        return samples
    # endregion PRIVATE

    # region OVERRIDE

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return self._n_samples
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        return self._samples[idx]
    # end __getitem__

    # endregion OVERRIDE

# end PeriodicSignalDataset
