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
class MatrixGenerator(object):
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

    # Get a parameter value
    def get_parameter(self, key):
        """
        Get a parameter value
        :param key: Parameter name
        :return: Parameter value
        """
        try:
            return self._parameters[key]
        except KeyError:
            raise Exception("Unknown parameter : {}".format(key))
        # end try
    # end get_parameter

    # Set a parameter value
    def set_parameter(self, key, value):
        """
        Set a parameter value
        :param key: Parameter name
        :param value: Parameter value
        """
        try:
            self._parameters[key] = value
        except KeyError:
            raise Exception("Unknown parameter : {}".format(key))
        # end try
    # end set_parameter

    # Generate the matrix
    def generate(self, size, dtype=torch.float64):
        """
        Generate the matrix
        :param size: Matrix size
        :param dtype: Data type
        :return: Generated matrix
        """
        return torch.randn(size)
    # end generate

    ################
    # PRIVATE
    ################

    # Set parameters
    def _set_parameters(self, args):
        """
        Set parameters
        :param args: Parameters as dict
        """
        for key, value in args.items():
            self.set_parameter(key, value)
        # end for
    # end _set_parameters

# end MatrixGenerator
