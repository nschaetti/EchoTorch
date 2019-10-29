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

        # Type
        if self.dtype == torch.float32:
            tensor_type = torch.FloatTensor
        else:
            tensor_type = torch.DoubleTensor
        # end if

        # Leak rate
        self.register_buffer('leaky_rate', Variable(tensor_type(1).fill_(leaky_rate), requires_grad=False))
    # end __init__

    ###############################################
    # PUBLIC
    ###############################################

    # Forward
    def forward(self, u, y=None, reset_state=True):
        """
        Forward
        :param u: Input signal.
        :param y: Training target (None if prediction)
        :param reset_state: Reset hidden state for each sample ?
        :return: Resulting hidden states.
        """
        # Time length
        time_length = int(u.size()[1])

        # Number of batches
        n_batches = int(u.size()[0])

        # Outputs
        outputs = Variable(torch.zeros(n_batches, time_length, self.output_dim, dtype=self.dtype))
        outputs = outputs.cuda() if self.hidden.is_cuda else outputs

        # For each batch
        for b in range(n_batches):
            # Reset hidden layer
            if reset_state:
                self.reset_hidden()
            # end if

                # Pre-update hook
                u[b, :] = self._pre_update_hook(u[b, :], b)

            # For each steps
            for t in range(time_length):
                # Current input
                ut = u[b, t]

                # Pre-hook
                ut = self._pre_step_update_hook(ut, t)

                # Compute input layer
                u_win = self.w_in.mv(ut)

                # Apply W to x
                x_w = self.w.mv(self.hidden)

                # Add everything
                x = u_win + x_w + self.w_bias

                # Apply activation function
                x = self.nonlin_func(x)

                # Leaky
                x = (self.hidden.mul(1.0 - self.leaky_rate) + x.view(self.output_dim).mul(self.leaky_rate))

                # Post-hook
                x = self._post_step_update_hook(x.view(self.output_dim), ut, t)

                # Add to outputs
                self.hidden.data = x.data

                # New last state
                outputs[b, t] = self.hidden
            # end for

            # Post-update hook
            outputs[b, :] = self._post_update_hook(outputs[b, :], u[b, :], b)
        # end for

        return outputs
    # end forward

    ###############################################
    # PRIVATE
    ###############################################

# end LiESNCell
