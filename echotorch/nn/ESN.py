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

import torch
import torch.nn as nn
from torch.autograd import Variable
from . import ESNCell


# Echo State Network module
class ESN(nn.Module):
    """
    Echo State Network module
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, output_dim, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0,
                 w=None, w_in=None, w_bias=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None,
                 nonlin_func=torch.tanh, learning_algo='inv', ridge_param=0.0, create_cell=True):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param hidden_dim: Hidden layer dimension
        :param output_dim: Reservoir size
        :param spectral_radius: Reservoir's spectral radius
        :param bias_scaling: Scaling of the bias, a constant input to each neuron (default: 0, no bias)
        :param input_scaling: Scaling of the input weight matrix, default 1.
        :param w: Internation weights matrix
        :param w_in: Input-reservoir weights matrix
        :param w_bias: Bias weights matrix
        :param sparsity:
        :param input_set:
        :param w_sparsity:
        :param nonlin_func: Reservoir's activation function (tanh, sig, relu)
        :param learning_algo: Which learning algorithm to use (inv, LU, grad)
        """
        super(ESN, self).__init__()

        # Properties
        self.output_dim = output_dim
        self.learning_algo = learning_algo
        self.ridge_param = ridge_param

        # Recurrent layer
        if create_cell:
            self.esn_cell = ESNCell(input_dim, hidden_dim, spectral_radius, bias_scaling, input_scaling, w, w_in, w_bias,
                                    sparsity, input_set, w_sparsity, nonlin_func)
        # end if

        # Linear layer if needed
        if learning_algo == 'grad':
            self.linear = nn.Linear(hidden_dim, output_dim, bias=True)
        else:
            # Set it as buffer
            self.register_buffer('xTx', Variable(torch.zeros(hidden_dim, hidden_dim), requires_grad=False))
            self.register_buffer('xTy', Variable(torch.zeros(hidden_dim, output_dim), requires_grad=False))
            self.register_buffer('w_out', Variable(torch.zeros(1, hidden_dim), requires_grad=False))
        # end if
    # end __init__

    ###############################################
    # PROPERTIES
    ###############################################

    # Hidden layer
    @property
    def hidden(self):
        """
        Hidden layer
        :return:
        """
        return self.esn_cell.hidden
    # end hidden

    # Hidden weight matrix
    @property
    def w(self):
        """
        Hidden weight matrix
        :return:
        """
        return self.esn_cell.w
    # end w

    ###############################################
    # PUBLIC
    ###############################################

    # Forward
    def forward(self, u, targets=None):
        """
        Forward
        :param u: Input signal.
        :param targets: Target outputs
        :return: Output or hidden states
        """
        # Batch size
        batch_size = u.size()[0]

        # Time length
        time_length = u.size()[1]

        # Compute hidden states
        hidden_states = self.esn_cell(u)

        # Learning algo
        if self.learning_algo != 'grad' and targets is not None:
            for b in range(batch_size):
                self.xTx.data.add_(hidden_states[b].t().mm(hidden_states[b]).data)
                self.xTy.data.add_(hidden_states[b].t().mm(targets[b].unsqueeze(1)).data)
            # end for
            return hidden_states
        elif self.learning_algo != 'grad':
            # Outputs
            outputs = Variable(torch.zeros(batch_size, time_length, self.output_dim), requires_grad=False)
            outputs = outputs.cuda() if self.hidden.is_cuda else outputs

            # For each batch
            for b in range(batch_size):
                outputs[b] = torch.mm(hidden_states[b], self.w_out)
            # end for

            return outputs
        else:
            # Linear output
            return self.linear(hidden_states)
        # end if
    # end forward

    # Finish training
    def finalize(self):
        """
        Finalize training with LU factorization
        """
        if self.learning_algo == 'inv':
            inv_xTx = self.xTx.inverse()
            self.w_out.data = torch.mm(inv_xTx, self.xTy).data
        else:
            self.w_out.data = torch.gesv(self.xTy, self.xTx + torch.eye(self.esn_cell.output_dim).mul(self.ridge_param)).data
        # end if
    # end finalize

    # Reset hidden layer
    def reset_hidden(self):
        """
        Reset hidden layer
        :return:
        """
        self.esn_cell.reset_hidden()
    # end reset_hidden

    # Get W's spectral radius
    def get_spectral_radius(self):
        """
        Get W's spectral radius
        :return: W's spectral radius
        """
        return self.esn_cell.get_spectral_raduis()
    # end spectral_radius

    ###############################################
    # PRIVATE
    ###############################################

# end ESNCell
