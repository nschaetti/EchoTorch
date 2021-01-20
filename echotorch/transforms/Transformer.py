# -*- coding: utf-8 -*-
#
# File : echotorch/transforms/ToOneHot.py
# Description : Transform integer targets to one-hot vectors.
# Date : 21th of November, 2019
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


# Base class for transformers
class Transformer(object):
    """
    Base class for transformers
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, input_dim, output_dim, time_dim=0, dtype=torch.float64):
        """
        Constructor
        """
        # Properties
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._time_dim = time_dim
        self._dtype = dtype
    # end __init__

    # endregion CONSTRUCTORS

    # region PROPERTIES

    # Dimension of the input timeseries
    @property
    def input_dim(self):
        """
        Dimension of the output timeseries
        :return: Dimension of the output timeseries
        """
        return self._input_dim
    # end output_dim

    # Dimension of the output timeseries
    @property
    def output_dim(self):
        """
        Dimension of the output timeseries
        :return: Dimension of the output timeseries
        """
        return self._output_dim
    # end output_dim

    # Position of the time dimension
    @property
    def time_dim(self):
        """
        Position of the time dimension
        :return: Position of the time dimension
        """
        return self._time_dim
    # end time_dim

    # Output type
    @property
    def dtype(self):
        """
        Output type
        :return: Output type
        """
        return self._dtype
    # end output_dim

    # endregion PROPERTIES

    # region PRIVATE

    # Transform
    def _transform(self, x):
        """
        Transform input
        :param x:
        :return:
        """
        return x
    # end _transform

    # endregion PRIVATE

    # region OVERRIDE

    # Convert a string
    def __call__(self, x):
        """
        Transform a time series
        :param x: Time series to transform
        :return: Transformed time series
        """
        return self._transform(x)
    # end convert

    # String
    def __repr__(self):
        """
        String
        :return:
        """
        # Class
        init_str = type(self).__name__ + "("

        # For each attributes
        index = 0
        for attr in dir(self):
            if "_" not in attr:
                attr_value = getattr(self, attr)
                if type(attr_value) is int or type(attr_value) is float or type(attr_value) is str\
                        or type(attr_value) is tuple:
                    add_begin = " " if index != 0 else ""
                    init_str += add_begin + "{}={}, ".format(attr, getattr(self, attr))
                    index += 1
                # end if
            # end if
        # end for

        # Remove ", "
        if init_str[-2:] == ", ":
            init_str = init_str[:-2]
        # end if

        # )
        init_str += ")"

        return init_str
    # end __str__

    # endregion OVERRIDE

    # region STATIC

    # endregion STATIC

# end Transformer
