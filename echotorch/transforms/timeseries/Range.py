# -*- coding: utf-8 -*-
#
# File : echotorch/transforms/timeseries/Normalize.py
# Description : Normalize a timeserie.
# Date : 12th of April, 2020
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
# Copyright Nils Schaetti <nils.schaetti@unine.ch>, <nils.schaetti@unige.ch>


# Imports
import torch
from ..Transformer import Transformer


# Range transformer
class Range(Transformer):
    """
    Put a timeseries in a range
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, input_dim, min, max, dtype=torch.float64):
        """
        Constructor
        :param input_dim: Input dimension
        :param min: Lower limit of the range
        :param max: Higher limit of the range
        """
        # Super constructor
        super(Range, self).__init__(input_dim=input_dim, output_dim=input_dim, dtype=dtype)

        # Properties
        self._min = min
        self._max = max
    # end __init__

    # endregion CONSTRUCTORS

    # region PROPERTIES

    # endregion PROPERTIES

    # region OVERRIDE

    # Transform
    def _transform(self, x):
        """
        Transform input
        :param x:
        :return:
        """
        # Maximum
        if isinstance(self._max, torch.Tensor):
            # For each channel
            for i_i in range(self._input_dim):
                x[x[:, i_i] > self._max[i_i], i_i] = self._max[i_i]
            # end for
        else:
            x[x > self._max] = self._max
        # end for

        # Minimum
        if isinstance(self._min, torch.Tensor):
            # For each channel
            for i_i in range(self._input_dim):
                x[x[:, i_i] < self._min[i_i], i_i] = self._min[i_i]
            # end for
        else:
            x[x < self._min] = self._min
        # end if

        return x
    # end _transform

    # endregion OVERRIDE

# end Normalize
