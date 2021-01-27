# -*- coding: utf-8 -*-
#
# File : echotorch/transforms/timeseries/FilterInfiniteValue.py
# Description : Filter infinite values in time series
# Date : 27th of January 2021
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


# FilterInfiniteValue
class FilterInfiniteValue(Transformer):
    """
    Filter infinite values in time series
    """

    # Constructor
    def __init__(self, input_dim, dummy_value, dtype=torch.float64):
        """
        Constructor
        """
        # Super constructor
        super(FilterInfiniteValue, self).__init__(
            input_dim=input_dim,
            output_dim=input_dim,
            dtype=dtype
        )

        # Properties
        self._dummy_value = dummy_value
    # end __init__

    # region PROPERTIES

    # Value to replace infinity
    @property
    def dummy_value(self):
        """
        Value to replace infinity
        :return: Value to replace infinity
        """
        return self._dummy_value
    # end dummy_value

    # endregion PROPERTIES

    # region OVERRIDE

    # Transform
    def _transform(self, x):
        """
        Transform input
        """
        x[torch.isinf(x)] = self._dummy_value
        return x
    # end _transform

    # endregion OVERRIDE

# end FilterInfiniteValue
