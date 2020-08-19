# -*- coding: utf-8 -*-
#
# File : echotorch/utils/matrix_factory.py
# Description : Matrix factor.
# Date : 29th of October, 2019
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
import torch.sparse


# Matrix factory
class MatrixFactory(object):
    """
    Matrix factory
    """

    # Instance
    _instance = None

    # Constructor
    def __init__(self):
        """
        Constructor
        """
        # Init. generators
        self._generators = {}

        # Save instance
        self._instance = self
    # end __init__

    #region PUBLIC

    # Register creator
    def register_generator(self, name, generator):
        """
        Register matrix generator
        :param name: Generator's name
        :param generator: Generator object
        """
        self._generators[name] = generator
    # end register_generator

    # Get a generator
    def get_generator(self, name, **kwargs):
        """
        Get a generator
        :param name: Generator's name
        :param kwargs: Arguments for the generator
        :return:
        """
        generator = self._generators[name]
        if not generator:
            raise ValueError(name)
        # end if
        return generator(**kwargs)
    # end get_generator

    #endregion PUBLIC

    #region STATIC

    # Get instance
    def get_instance(self):
        """
        Get instance
        :return: Instance
        """
        return self._instance
    # end get_instance

    # To sparse matrix
    @staticmethod
    def to_sparse(m):
        """
        To sparse matrix
        :param m:
        :return:
        """
        # Rows, columns and values
        rows = torch.LongTensor()
        columns = torch.LongTensor()
        values = torch.FloatTensor()

        # For each row
        for i in range(m.shape[0]):
            # For each column
            for j in range(m.shape[1]):
                if m[i, j] != 0.0:
                    rows = torch.cat((rows, torch.LongTensor([i])), dim=0)
                    columns = torch.cat((columns, torch.LongTensor([j])), dim=0)
                    values = torch.cat((values, torch.FloatTensor([m[i, j]])), dim=0)
                # end if
            # end for
        # end for

        # Indices
        indices = torch.cat((rows.unsqueeze(0), columns.unsqueeze(0)), dim=0)

        # To sparse
        return torch.sparse.FloatTensor(indices, values)
    # end to_sparse

    #endregion STATIC

# end MatrixFactory


# Create factory
matrix_factory = MatrixFactory()
