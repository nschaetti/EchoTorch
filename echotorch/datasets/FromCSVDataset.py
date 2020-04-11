# -*- coding: utf-8 -*-
#
# File : datasets/FromCSVDataset.py
# Description : Load timeseries from a CSV file.
# Date : 10th of April, 2020
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
import torch.distributions.multinomial
from torch.utils.data.dataset import Dataset
import numpy as np


# Load Time series from a CSV file
class FromCSVDataset(Dataset):
    """
    Load Time series from a CSV file
    """

    # Constructor
    def __init__(self, csv_file, columns, *args, **kwargs):
        """
        Constructor
        :param csv_file: CSV file
        :param columns: Columns to load from the CSV file
        :param args: Args
        :param kwargs: Dictionary args
        """
        # Super
        super(FromCSVDataset, self).__init__(*args, **kwargs)

        # Properties
        self._csv_file = csv_file
        self._columns = columns

        # Load
        self._data = self._load_from_csv(self._csv_file, self._columns)
    # end __init__

    # region PRIVATE

    # Load from CSV file
    def _load_from_csv(self, csv_file, columns):
        """
        Load from CSV file
        :param csv_file: CSV file
        :param columns: Columns
        :return:
        """
        pass
    # end _load_from_csv

    # endregion ENDPRIVATE

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
        :param idx: Sample index
        :return: Sample as torch tensor
        """
        # Generate a Markov chain with
        # specified length.
        return self._generate_markov_chain(
            length=self._sample_length,
            start_state=np.random.randint(low=0, high=self._n_states-1)
        )
    # end __getitem__

    # endregion OVERRIDE

# end DiscreteMarkovChainDataset
