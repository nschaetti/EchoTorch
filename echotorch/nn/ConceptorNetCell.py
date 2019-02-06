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
from torch.autograd import Variable
from .LiESNCell import LiESNCell
from .Conceptor import Conceptor
from .ConceptorPool import ConceptorPool


# Special reservoir layer for Conceptors
class ConceptorNetCell(LiESNCell):
    """
    Special reservoir layer for Conceptors
    """

    # Constructor
    def __init__(self, *args, **kwargs):
        """
        Constructor
        :param leaky_rate: Reservoir's leaky rate (default 1.0, normal ESN)
        :param train_leaky_rate: Train leaky rate as parameter? (default: False)
        """
        super(ConceptorNetCell, self).__init__(*args, **kwargs)
    # end __init__

    ###############################################
    # PUBLIC
    ###############################################

    # Forward
    def forward(self, u=None, y=None, w_out=None, reset_state=True, input_recreation=None, conceptor=None, length=None, mu=None):
        """
        Forward execution
        :param u:
        :param y:
        :param w_out:
        :param reset_state:
        :param generative_mode:
        :return:
        """
        # Time length
        if u is not None:
            time_length = int(u.size()[1])
        else:
            time_length = length
        # end if

        # Number of batches
        if u is not None:
            n_batches = int(u.size()[0])
        else:
            n_batches = 1
        # end if

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
                # Generative mode ?
                if self.training:
                    # Current input
                    ut = u[b, t]

                    # Compute input layer
                    u_win = self.w_in.mv(ut)

                    # Apply W to x
                    x_w = self.w.mv(self.hidden)

                    # Add everything
                    x = u_win + x_w + self.w_bias

                    # Apply activation function
                    x = self.nonlin_func(x)

                    # Add to outputs
                    self.hidden.data = x.view(-1).data

                    # New last state
                    outputs[b, t] = self.hidden
                else:
                    if u is not None:
                        # Current input
                        ut = u[b, t]

                        # Compute input layer
                        u_win = self.w_in.mv(ut)

                        # Apply W to x
                        x_w = self.w.mv(self.hidden)

                        # Add everything
                        x = u_win + x_w + self.w_bias
                    else:
                        # Apply W to x
                        x_w = input_recreation(self.hidden.view(1, 1, -1))

                        # Add everything
                        x = x_w + self.w_bias
                    # end if

                    # Apply activation function
                    x = self.nonlin_func(x)

                    # Apply conceptor
                    if type(conceptor) is Conceptor:
                        xc = conceptor(x.view(1, 1, -1)).view(-1)
                    elif type(conceptor) is ConceptorPool:
                        # Apply morphing
                        M = conceptor.morphing(mu[b, t])
                        xc = M(x.view(1, 1, -1)).view(-1)
                    else:
                        xc = x.view(-1)
                    # end if

                    # New hidden
                    self.hidden.data = xc.data

                    # Add to outputs
                    outputs[b, t] = self.hidden
                # end if
            # end for
        # end for

        return outputs
    # end forward

    ###############################################
    # PRIVATE
    ###############################################

# end LiESNCell
