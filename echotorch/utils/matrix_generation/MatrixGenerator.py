# -*- coding: utf-8 -*-
#
# File : echotorch/utils/matrix_generation/MatrixGenerator.py
# Description : Matrix generator base class.
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

# Import
import torch


# Matrix generator base object
class MatrixGenerator:
    """
    Matrix generator base object
    """

    # Constructor
    def __init__(self, **kwargs):
        """
        Constructor
        :param kwargs: Parameters of the generator
        """
        self._parameters = kwargs
    # end __init__

    ################
    # PROPERTIES
    ################

    # Parameters
    @property
    def parameters(self):
        """
        Parameters
        :return: Generator's parameter
        """
        return self._parameters
    # end parameters

    ################
    # PUBLIC
    ################

    # Generate the matrix
    def generate(self, size, dtype=torch.float32):
        """
        Generate the matrix
        :param size: Matrix size
        :param dtype: Data type
        :return: Generated matrix
        """
        return torch.randn(size)
    # end generate

# end MatrixGenerator
