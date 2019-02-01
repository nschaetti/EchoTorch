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
from .ESNCell import ESNCell
import matplotlib.pyplot as plt


# Leak-Integrated Echo State Network layer
class LiESNCell(ESNCell):
    """
    Leaky-Integrated Echo State Network layer
    """

    # Constructor
    def __init__(self, leaky_rate=1.0, train_leaky_rate=False, *args, **kwargs):
        """
        Constructor
        :param leaky_rate: Reservoir's leaky rate (default 1.0, normal ESN)
        :param train_leaky_rate: Train leaky rate as parameter? (default: False)
        """
        super(LiESNCell, self).__init__(*args, **kwargs)

        # Type
        if self.dtype == torch.float32:
            tensor_type = torch.FloatTensor
        else:
            tensor_type = torch.DoubleTensor
        # end if

        # Params
        if train_leaky_rate:
            self.leaky_rate = nn.Parameter(tensor_type(1).fill_(leaky_rate), requires_grad=True)
        else:
            # Initialize bias
            self.register_buffer('leaky_rate', Variable(tensor_type(1).fill_(leaky_rate), requires_grad=False))
        # end if
    # end __init__

    ###############################################
    # PUBLIC
    ###############################################

    # Forward
    def forward(self, u, y=None, w_out=None, reset_state=True):
        """
        Forward
        :param u: Input signal.
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

            # For each steps
            for t in range(time_length):
                # Current input
                ut = u[b, t]

                # Compute input layer
                u_win = self.w_in.mv(ut)

                # Apply W to x
                x_w = self.w.mv(self.hidden)

                # Feedback or not
                if self.feedbacks and self.training and y is not None:
                    # Current target
                    yt = y[b, t]

                    # Compute feedback layer
                    y_wfdb = self.w_fdb.mv(yt)

                    # Add everything
                    x = u_win + x_w + y_wfdb + self.w_bias
                elif self.feedbacks and not self.training and w_out is not None:
                    # Add bias
                    bias_hidden = torch.cat((Variable(torch.ones(1)), self.hidden), dim=0)

                    # Compute past output
                    yt = w_out.t().mv(bias_hidden)

                    # Normalize
                    if self.normalize_feedbacks:
                        yt -= torch.min(yt)
                        yt /= torch.max(yt) - torch.min(yt)
                        yt /= torch.sum(yt)
                    # end if

                    # Compute feedback layer
                    y_wfdb = self.w_fdb.mv(yt)

                    # Add everything
                    x = u_win + x_w + y_wfdb + self.w_bias
                else:
                    # Add everything
                    x = u_win + x_w + self.w_bias
                # end if

                # Apply activation function
                x = self.nonlin_func(x)

                # Add to outputs
                self.hidden.data = (self.hidden.mul(1.0 - self.leaky_rate) + x.view(self.output_dim).mul(self.leaky_rate)).data

                # New last state
                outputs[b, t] = self.hidden
            # end for
        # end for

        return outputs
    # end forward

    ###############################################
    # PRIVATE
    ###############################################

# end LiESNCell
