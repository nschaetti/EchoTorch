# -*- coding: utf-8 -*-
#
# File : echotorch/transforms/timeseries/Scale.py
# Description : Multiply channels by a constant
# Date : 26th of January, 2021
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


# Scale transformer
class Scale(Transformer):
    """
    Multiply channels by a constant
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, input_dim, scales, dtype=torch.float64):
        """
        Constructor
        :param input_dim: Input dimension
        :param scales: Scales as a scalar or a tensor
        """
        # Super constructor
        super(Scale, self).__init__(input_dim=input_dim, output_dim=input_dim, dtype=dtype)

        # Properties
        self._scales = scales
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
        if isinstance(self._scales, torch.Tensor):
            return torch.mm(x, torch.diag(self._scales))
        else:
            return x * self._scales
        # end if
    # end _transform

    # endregion OVERRIDE

# end Scale
