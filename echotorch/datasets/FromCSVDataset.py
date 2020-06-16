# -*- coding: utf-8 -*-
#
# File : datasets/FromCSVDataset.py
# Description : Load time series from a CSV file.
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
import torch
from torch.utils.data.dataset import Dataset
import csv


# Load Time series from a CSV file
class FromCSVDataset(Dataset):
    """
    Load Time series from a CSV file
    """

    # Constructor
    def __init__(self, csv_file, columns, delimiter=",", quotechar='"', *args, **kwargs):
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
        self._n_columns = len(columns)
        self._delimiter = delimiter
        self._quotechar = quotechar
        self._column_indices = list()

        # Load
        self._data = self._load_from_csv()
    # end __init__

    # region PRIVATE

    # Find indices for each column
    def _find_columns_indices(self, header_row):
        """
        Find indices for each column
        :param header_row: Header row
        """
        for col in self._columns:
            if col in header_row:
                self._column_indices.append(header_row.index(col))
            else:
                raise Exception("Not column \'{}\' found in the CSV".format(col))
            # end if
        # end for
    # end _find_columns_indices

    # Load from CSV file
    def _load_from_csv(self):
        """
        Load from CSV file
        :return:
        """
        # Open CSV file
        with open(self._csv_file, 'r') as csvfile:
            # Read data
            spamreader = csv.reader(csvfile, delimiter=self._delimiter, quotechar=self._quotechar)

            # Data
            data = list()

            # For each row
            for row_i, row in enumerate(spamreader):
                # First row is the column name
                if row_i == 0:
                    self._find_columns_indices(row)
                else:
                    # Row list
                    row_list = list()

                    # Add each column
                    for idx in self._column_indices:
                        row_list.append(row[idx])
                    # end for

                    # Add to data
                    data.append(row_list)
                # end if
            # end for

            # Create tensor
            data_tensor = torch.zeros(len(data), self._n_columns)

            # Insert data in tensor
            for row_i, row in enumerate(data):
                for col_i, e in enumerate(row):
                    data_tensor[row_i, col_i] = float(e)
                # end for
            # end for

            return data_tensor
        # end for
    # end _load_from_csv

    # endregion ENDPRIVATE

    # region OVERRIDE

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return 1
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
        return self._data
    # end __getitem__

    # endregion OVERRIDE

# end DiscreteMarkovChainDataset
