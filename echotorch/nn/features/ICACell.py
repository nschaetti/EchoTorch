# -*- coding: utf-8 -*-
#
# File : echotorch/nn/ESN.py
# Description : An Echo State Network module.
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

# Imports
import torch.sparse
import torch
import torch.nn as nn
from torch.autograd import Variable


# Independent Component Analysis layer
class ICACell(nn.Module):
    """
    Principal Component Analysis layer. It can be used to handle different batch-mode algorithm for ICA.
    """

    # Constructor
    def __init__(self, input_dim, output_dim):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param output_dim: Reservoir size
        """
        super(ICACell, self).__init__()
        pass
    # end __init__

    # region PROPERTIES

    # endregion PROPERTIES

    # region PRIVATE

    # Add constant
    def _add_constant(self, x):
        """
        Add constant
        :param x:
        :return:
        """
        bias = Variable(torch.ones((x.size()[0], x.size()[1], 1)), requires_grad=False)
        return torch.cat((bias, x), dim=2)
    # end _add_constant

    # endregion PRIVATE

    # region OVERRIDE

    # Reset learning
    def reset(self):
        """
        Reset learning
        :return:
        """
        # Training mode again
        self.train(True)
    # end reset

    # Forward
    def forward(self, x, y=None):
        """
        Forward
        :param x: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        # Batch size
        batch_size = x.size()[0]

        # Time length
        time_length = x.size()[1]

        # Add bias
        if self.with_bias:
            x = self._add_constant(x)
        # end if
    # end forward

    # Finish training
    def finalize(self):
        """
        Finalize training with LU factorization or Pseudo-inverse
        """
        pass
    # end finalize

    # endregion OVERRIDE

# end ICACell
