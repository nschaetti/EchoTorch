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
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>

"""
Created on 26 January 2018
@author: Nils Schaetti
"""

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from echotorch.nn.reservoir.LiESNCell import LiESNCell
from echotorch.nn.features.PCACell import PCACell
from torch.autograd import Variable


# Gated Echo State Network
class GatedESN(nn.Module):
    """
    Gated Echo State Network
    """

    # Constructor
    def __init__(self, input_dim, reservoir_dim, pca_dim, hidden_dim, leaky_rate=1.0, spectral_radius=0.9,
                 bias_scaling=0, input_scaling=1.0, w=None, w_in=None, w_bias=None, sparsity=None,
                 input_set=[1.0, -1.0], w_sparsity=None, nonlin_func=torch.tanh,
                 create_cell=True):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param hidden_dim: Hidden layer dimension
        :param reservoir_dim: Reservoir size
        :param spectral_radius: Reservoir's spectral radius
        :param bias_scaling: Scaling of the bias, a constant input to each neuron (default: 0, no bias)
        :param input_scaling: Scaling of the input weight matrix, default 1.
        :param w: Internal weights matrix
        :param w_in: Input-reservoir weights matrix
        :param w_bias: Bias weights matrix
        :param sparsity:
        :param input_set:
        :param w_sparsity:
        :param nonlin_func: Reservoir's activation function (tanh, sig, relu)
        :param learning_algo: Which learning algorithm to use (inv, LU, grad)
        """
        super(GatedESN, self).__init__()

        # Properties
        self.reservoir_dim = reservoir_dim
        self.pca_dim = pca_dim
        self.hidden_dim = hidden_dim
        self.finalized = False

        # Recurrent layer
        if create_cell:
            self.esn_cell = LiESNCell(
                input_dim=input_dim, output_dim=reservoir_dim, spectral_radius=spectral_radius, bias_scaling=bias_scaling,
                input_scaling=input_scaling, w=w, w_in=w_in, w_bias=w_bias, sparsity=sparsity, input_set=input_set,
                w_sparsity=w_sparsity, nonlin_func=nonlin_func, leaky_rate=leaky_rate
            )
        # end if

        # PCA
        if self.pca_dim > 0:
            self.pca_cell = PCACell(input_dim=reservoir_dim, output_dim=pca_dim)
        # end if

        # Initialize input update weights
        self.register_parameter('wzp', nn.Parameter(self.init_wzp()))

        # Initialize hidden update weights
        self.register_parameter('wzh', nn.Parameter(self.init_wzh()))

        # Initialize update bias
        self.register_parameter('bz', nn.Parameter(self.init_bz()))
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

    # Init hidden vector
    def init_hidden(self):
        """
        Init hidden layer
        :return: Initiated hidden layer
        """
        return Variable(torch.zeros(self.hidden_dim), requires_grad=False)
    # end init_hidden

    # Init update vector
    def init_update(self):
        """
        Init hidden layer
        :return: Initiated hidden layer
        """
        return self.init_hidden()
    # end init_hidden

    # Init update-reduced matrix
    def init_wzp(self):
        """
        Init update-reduced matrix
        :return: Initiated update-reduced matrix
        """
        return torch.rand(self.pca_dim, self.hidden_dim)
    # end init_hidden

    # Init update-hidden matrix
    def init_wzh(self):
        """
        Init update-hidden matrix
        :return: Initiated update-hidden matrix
        """
        return torch.rand(self.pca_dim, self.hidden_dim)
    # end init_hidden

    # Init update bias
    def init_bz(self):
        """
        Init update bias
        :return:
        """
        return torch.rand(self.hidden_dim)
    # end init_bz

    # Reset learning
    def reset(self):
        """
        Reset learning
        :return:
        """
        # Reset PCA layer
        self.pca_cell.reset()

        # Reset reservoir
        self.reset_reservoir()

        # Training mode again
        self.train(True)
    # end reset

    # Forward
    def forward(self, u, y=None):
        """
        Forward
        :param u: Input signal.
        :return: Output or hidden states
        """
        # Time length
        time_length = int(u.size()[1])

        # Number of batches
        n_batches = int(u.size()[0])

        # Compute reservoir states
        reservoir_states = self.esn_cell(u)
        reservoir_states.required_grad = False

        # Reduce
        if self.pca_dim > 0:
            # Reduce states
            pca_states = self.pca_cell(reservoir_states)
            pca_states.required_grad = False

            # Stop here if we learn PCA
            if self.finalized:
                return
            # end if

            # Hidden states
            hidden_states = Variable(torch.zeros(n_batches, time_length, self.hidden_dim))
            hidden_states = hidden_states.cuda() if pca_states.is_cuda else hidden_states
        else:
            # Hidden states
            hidden_states = Variable(torch.zeros(n_batches, time_length, self.hidden_dim))
            hidden_states = hidden_states.cuda() if reservoir_states.is_cuda else hidden_states
        # end if

        # For each batch
        for b in range(n_batches):
            # Reset hidden layer
            hidden = self.init_hidden()

            # TO CUDA
            if u.is_cuda:
                hidden = hidden.cuda()
            # end if

            # For each steps
            for t in range(time_length):
                # Current reduced state
                if self.pca_dim > 0:
                    pt = pca_states[b, t]
                else:
                    pt = reservoir_states[b, t]
                # end if

                # Compute update vector
                zt = F.sigmoid(self.wzp.mv(pt) + self.wzh.mv(hidden) + self.bz)

                # Compute hidden state
                ht = (1.0 - zt) * hidden + zt * pt

                # Add to outputs
                hidden = ht.view(self.hidden_dim)

                # New last state
                hidden_states[b, t] = hidden
            # end for
        # end for

        # Return hidden states
        return hidden_states
    # end forward

    # Finish training
    def finalize(self):
        """
        Finalize training with LU factorization
        """
        # Finalize output training
        self.pca_cell.finalize()

        # Finalized
        self.finalized = True
    # end finalize

    # Reset reservoir layer
    def reset_reservoir(self):
        """
        Reset hidden layer
        :return:
        """
        self.esn_cell.reset_hidden()
    # end reset_reservoir

    # Reset hidden layer
    def reset_hidden(self):
        """
        Reset hidden layer
        :return:
        """
        self.hidden.fill_(0.0)
    # end reset_hidden

# end GatedESN
