# -*- coding: utf-8 -*-
#
# File : echotorch/nn/LiESNCell.py
# Description : An Leaky-Integrated Echo State Network layer.
# Date : 26th of January, 2018
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

"""
Created on 26 January 2018
@author: Nils Schaetti
"""

import torch
import torch.sparse
import torch.nn as nn
from torch.autograd import Variable
from echotorch.nn.reservoir.ESNCell import ESNCell


# Leak-Integrated Echo State Network layer
class LiESNCell(ESNCell):
    """
    Leaky-Integrated Echo State Network layer
    """

    # Constructor
    def __init__(self, leaky_rate=1.0, *args, **kwargs):
        """
        Constructor
        :param leaky_rate: Reservoir's leaky rate (default 1.0, normal ESN)
        """
        super(LiESNCell, self).__init__(*args, **kwargs)

        # Param
        self._leaky_rate = leaky_rate
    # end __init__

    # region OVERRIDE

    # Compute post nonlinearity hook
    def _post_nonlinearity(self, x):
        """
        Compute post nonlinearity hook
        :param x: Reservoir state at time t
        :return: Reservoir state
        """
        return self.hidden.mul(1.0 - self._leaky_rate) + x.view(self.output_dim).mul(self._leaky_rate)
    # end _post_nonlinearity

    # Extra-information
    def extra_repr(self):
        """
        Extra-information
        """
        s = super(LiESNCell, self).extra_repr()
        s += ', leaky-rate={_leaky_rate}'
        return s.format(**self.__dict__)
    # end extra_repr

    # endregion OVERRIDE

# end LiESNCell
