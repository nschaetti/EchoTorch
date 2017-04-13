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

    def __init__(self, input_features, output_features, size, bias=True, initial_state=None):
        """
        Constructor
        :param input_features: Number of input features.
        :param reservoir_features:  Reservoir's size.
        :param output_features: Number of outputs
        :param size: Reservoir size
        :param bias: Use bias?
        """
        # Params
        self.input_features = input_features
        self.output_features = output_features
        self.size = size
        self.bias = bias
        self.initial_state = initial_state

        # The learnable output weights
        self.weight = nn.Parameter(torch.Tensor(output_features))

        # Initialize reservoir vector
        if self.initial_state is not None:
            self.x = Variable(self.initial_state, requires_grad=False)
        else:
            self.x = Variable(torch.zeros(self.size), requires_grad=False)
        # end if

        # Initialize inout weights
        self.win = Variable(torch.rand(self.size, self.input_features), requires_grad=False)

        # Initialize reservoir weights randomly
        self.w = Variable(torch.rand(self.size, self.size), requires_grad=False)

        # Linear output
        self.ll = nn.Linear(self.size, self.output_features)

    # end __init__


    # Forward
    def forward(self, u, x):
        """
        Forward
        :param u: Input signal
        :return: I don't know.
        """
        print("u : ")
        print(u)
        print("x : ")
        print(x)
        #x = F.tanh(self.win.mv(u) + self.w.mv(self.x))
        p = self.ll(x)

        return p, x
    # end forward

# end Reservoir