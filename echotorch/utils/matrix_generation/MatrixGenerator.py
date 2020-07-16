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
import echotorch.utils
import warnings


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
        # Default generation parameters
        self._parameters = dict()
        self._parameters['spectral_radius'] = 0.99
        self._parameters['apply_spectral_radius'] = True
        self._parameters['scale'] = 1.0

        # Set parameter values given
        for key, value in kwargs.items():
            self._parameters[key] = value
        # end for
    # end __init__

    #region PROPERTIES

    # Parameters
    @property
    def parameters(self):
        """
        Parameters
        :return: Generator's parameter
        """
        return self._parameters
    # end parameters

    #endregion PROPERTIES

    #region PUBLIC

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
        # Call matrix generation function
        w = self._generate_matrix(size, dtype)

        # Scale
        w *= self.get_parameter('scale')

        # Set spectral radius
        # If two dim tensor, square matrix and spectral radius is available
        if w.ndimension() == 2 and w.size(0) == w.size(1) and self.get_parameter('apply_spectral_radius'):
            # If current spectral radius is not zero
            if echotorch.utils.spectral_radius(w) > 0.0:
                w = (w / echotorch.utils.spectral_radius(w)) * self.get_parameter('spectral_radius')
            else:
                warnings.warn("Spectral radius of W is zero (due to small size), spectral radius not changed")
            # end if
        # end if

        return w
    # end generate

    #endregion PUBLIC

    #region PRIVATE

    # Generate the matrix
    def _generate_matrix(self, size, dtype=torch.float64):
        """
        Generate the matrix
        :param size: Matrix size
        :param dtype: Matrix data type
        :return: Generated matrix
        """
        return  torch.randn(size, dtype=dtype)
    # end _generate_matrix

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

    #endregion PRIVATE

# end MatrixGenerator
