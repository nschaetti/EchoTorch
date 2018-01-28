# -*- coding: utf-8 -*-
#
# File : echotorch/nn/ESNCell.py
# Description : An Echo State Network layer.
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
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import EchoTorch.tools
import numpy as np
import EchoTorch.nn.ESNCell


# Leak-Integrated Echo State Network layer
class LiESNCell(EchoTorch.nn.ESNCell):
    """
    Leaky-Integrated Echo State Network layer
    """

    # Constructor
    def __init__(self, leaky_rate=1.0, train_leaky_rate=False, **kwargs):
        """
        Constructor
        :param leaky_rate: Reservoir's leaky rate (default 1.0, normal ESN)
        :param train_leaky_rate: Train leaky rate as parameter? (default: False)
        """
        super(LiESNCell, self).__init__(kwargs)

        # Params
        self.leaky_rate = Variable(torch.Tensor(leaky_rate), requires_grad=train_leaky_rate)
    # end __init__

    ###############################################
    # PUBLIC
    ###############################################

    # Forward
    def forward(self, u, hidden):
        """
        Forward
        :param u: Input signal.
        :param x: Hidden layer state (x).
        :return: Resulting hidden states.
        """
        # Steps
        steps = int(u.size()[0])

        # Outputs
        outputs = Variable(torch.zeros(steps, self.output_dim))

        # For each steps
        for i in range(steps):
            # Current input
            ut = u[i]

            # Compute input layer
            u_win = self.w_in.mv(ut)

            # Apply W to x
            x_w = self.w.mv(hidden)

            # Apply activation function
            x_w = self.nonlin_func(x_w)

            # Add everything
            x = u_win + x_w + self.w_bias

            # Add to outputs
            hidden = x.view(self.output_dim)

            # New last state
            outputs[i] = hidden
        # end for

        return outputs
    # end forward

    ###############################################
    # PRIVATE
    ###############################################

# end ESNCell
