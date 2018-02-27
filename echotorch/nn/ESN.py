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
from . import ESNCell


# Echo State Network module
class ESN(nn.Module):
    """
    Echo State Network module
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, output_dim, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0,
                 w=None, w_in=None, w_bias=None, w_fdb=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None,
                 nonlin_func=torch.tanh, learning_algo='inv', ridge_param=0.0, create_cell=True,
                 feedbacks=False, with_bias=True, wfdb_sparsity=None, normalize_feedbacks=False):
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
        :param w_fdb: Feedback weights matrix
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
        self.feedbacks = feedbacks
        self.with_bias = with_bias
        self.normalize_feedbacks = normalize_feedbacks

        # Recurrent layer
        if create_cell:
            self.esn_cell = ESNCell(input_dim, hidden_dim, spectral_radius, bias_scaling, input_scaling, w, w_in,
                                    w_bias, w_fdb, sparsity, input_set, w_sparsity, nonlin_func, feedbacks, output_dim,
                                    wfdb_sparsity, normalize_feedbacks)
        # end if

        # Linear layer if needed
        if learning_algo == 'grad':
            self.linear = nn.Linear(hidden_dim, output_dim, bias=True)
        else:
            # Size
            if self.with_bias:
                self.x_size = hidden_dim + 1
            else:
                self.x_size = hidden_dim
            # end if

            # Set it as buffer
            self.register_buffer('xTx', Variable(torch.zeros(self.x_size, self.x_size), requires_grad=False))
            self.register_buffer('xTy', Variable(torch.zeros(self.x_size, output_dim), requires_grad=False))
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

    # Input matrix
    @property
    def w_in(self):
        """
        Input matrix
        :return:
        """
        return self.esn_cell.w_in
    # end w_in

    ###############################################
    # PUBLIC
    ###############################################

    # Reset learning
    def reset(self):
        """
        Reset learning
        :return:
        """
        if self.learning_algo == 'grad':
            self.linear.reset_parameters()
        else:
            self.xTx.data = torch.zeros(self.x_size, self.x_size)
            self.xTy.data = torch.zeros(self.x_size, self.output_dim)
            self.w_out.data = torch.zeros(1, self.output_dim)
        # end if

        # Training mode again
        self.train(True)
    # end reset

    # Output matrix
    def get_w_out(self):
        """
        Output matrix
        :return:
        """
        if self.learning_algo == 'grad':
            return self.linear.weight
        else:
            return self.w_out
        # end if
    # end get_w_out

    # Set W
    def set_w(self, w):
        """
        Set W
        :param w:
        :return:
        """
        self.esn_cell.w = w
    # end set_w

    # Forward
    def forward(self, u, y=None):
        """
        Forward
        :param u: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        # Batch size
        batch_size = u.size()[0]

        # Time length
        time_length = u.size()[1]

        # Compute hidden states
        if self.feedbacks and self.training:
            hidden_states = self.esn_cell(u, y)
        elif self.feedbacks and not self.training:
            hidden_states = self.esn_cell(u, w_out=self.w_out)
        else:
            hidden_states = self.esn_cell(u)
        # end if

        # Add bias
        if self.with_bias:
            hidden_states = self._add_constant(hidden_states)
        # end if

        # Learning algo
        if self.learning_algo != 'grad' and self.training:
            for b in range(batch_size):
                self.xTx.data.add_(hidden_states[b].t().mm(hidden_states[b]).data)
                self.xTy.data.add_(hidden_states[b].t().mm(y[b]).data)
            # end for
            return hidden_states
        elif self.learning_algo != 'grad' and not self.training:
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

        # Not in training mode anymore
        self.train(False)
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

# end ESNCell
