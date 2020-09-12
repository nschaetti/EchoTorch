# -*- coding: utf-8 -*-
#
# File : echotorch/transforms/timeseries/ToOneHot.py
# Description : Transform a series of integers to one-hot vectors.
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
# Copyright Nils Schaetti <nils.schaetti@unine.ch>

# Imports
import torch
from ..Transformer import Transformer


# Transform a series of integers to one-hot vectors
class ToOneHot(Transformer):
    """
    Transform a series of integers to one-hot vectors
    """

    # Constructor
    def __init__(self, output_dim, dtype=torch.float64):
        """
        Constructor
        """
        # Super constructor
        super(ToOneHot, self).__init__(
            input_dim=1,
            output_dim=output_dim
        )

        # Properties
        self._dtype = dtype
    # end __init__

    #region PRIVATE

    # Transform
    def _transform(self, x):
        """
        Transform input
        :param x:
        :return:
        """
        # The output timeseries
        output_series = torch.zeros(x.size(0), self.output_dim, dtype=self._dtype)

        # Transform each symbol
        for t in range(x.size(0)):
            output_series[t, x[t]] = 1.0
        # end for

        return output_series
    # end _transform

    #endregion PRIVATE

# end ToOneHot
