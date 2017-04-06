# -*- coding: utf-8 -*-
#
# File : echotorch/nn/LeakyReservoir.py
# Description : Implement the Echo State Network neural network with leaky units.
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
"""
Created on 6 April 2017
@author: Nils Schaetti
"""

import torch
import torch.nn as nn


# Echo State Network Reservoir module with leaky units.
class LeakyReservoir(nn.Module):
    """
    Echo State Network Reservoir module with leaky units.
    """

    def __init__(self, input_features, reservoir_features, output_features, bias=True):
        """
        Constructor
        :param input_features: Number of input features.
        :param reservoir_features:  Reservoir's size.
        :param output_features: Number of outputs
        :param bias: Use bias?
        """
        # Params
        self.input_features = input_features
        self.reservoir_features = reservoir_features
        self.output_features = output_features

        # Parameters
        self.weight = nn.Parameter(torch.Tensor(output_features))

        # If bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(reservoir_features))
        else:
            self.register_parameter('bias', None)
        # end if

        # Initialize reservoir weights randomly
        self.w = torch.Tensor.random_(-1, 2)

    # end __init__


    # Forward
    def forward(self, input):
        """
        Forward
        :param input: Input
        :return: I don't know.
        """
        return LeakyReservoir()(input, self.weight, self.bias)
    # end forward

# end Reservoir
