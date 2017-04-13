# -*- coding: utf-8 -*-
#
# File : echotorch/nn/Reservoir.py
# Description : Implement the Echo State Network neural network.
# Date : 6th of April, 2017
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
Created on 6 April 2017
@author: Nils Schaetti
"""

import torch
from torch.autograd import Variable
import torch.nn as nn


# Echo State Network Reservoir module
class Reservoir(nn.Module):
    """
    Echo State Network Reservoir module
    """

    def __init__(self, size, input_features, reservoir_features, output_features, bias=True):
        """
        Constructor
        :param input_features: Number of input features.
        :param reservoir_features:  Reservoir's size.
        :param output_features: Number of outputs
        :param bias: Use bias?
        """
        # Params
        self.size = size
        self.input_features = input_features
        self.reservoir_features = reservoir_features
        self.output_features = output_features

        # Parameters
        """self.weight = nn.Parameter(torch.Tensor(output_features))

        # If bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(reservoir_features))
        else:
            self.register_parameter('bias', None)
        # end if"""

        # Initialize inout weights
        #self.win = Variable(torch.rand(input_features), )

        # Initialize reservoir weights randomly
        #self.w = Variable(torch.rand(size, size), requires_grad=False)

    # end __init__


    # Forward
    def forward(self, u):
        """
        Forward
        :param input: Input
        :return: I don't know.
        """

        return u
    # end forward

# end Reservoir