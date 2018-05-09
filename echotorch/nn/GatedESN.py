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
from .BDESNCell import BDESNCell
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
from torch.autograd import Variable


# Gated Echo State Network
class GatedESN(nn.Module):
    """
    Gated Echo State Network
    """

    # Constructor
    def __init__(self, input_dim, reservoir_dim, pca_dim, hidden_dim, leaky_rate=1.0, spectral_radius=0.9,
                 bias_scaling=0, input_scaling=1.0, w=None, w_in=None, w_bias=None, sparsity=None,
                 input_set=[1.0, -1.0], w_sparsity=None, nonlin_func=torch.tanh, learning_algo='inv', ridge_param=0.0,
                 create_cell=True, pca_batch_size=10):
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

        # Recurrent layer
        if create_cell:
            self.esn_cell = BDESNCell(
                input_dim=input_dim, hidden_dim=reservoir_dim, spectral_radius=spectral_radius, bias_scaling=bias_scaling,
                input_scaling=input_scaling, w=w, w_in=w_in, w_bias=w_bias, sparsity=sparsity, input_set=input_set,
                w_sparsity=w_sparsity, nonlin_func=nonlin_func, leaky_rate=leaky_rate, create_cell=create_cell
            )
        # end if

        # PCA
        if self.pca_dim > 0:
            self.ipca = IncrementalPCA(n_components=pca_dim, batch_size=pca_batch_size)
        # end if

        # Init hidden vector
        self.register_buffer('hidden', self.init_hidden())

        # Init update vector
        self.register_buffer('update', self.init_update())

        # Initialize input update weights
        self.register_buffer('wzp', self._init_wzp())

        # Initialize hidden update weights
        self.register_buffer('wzh', self._init_wzh())

        # Initialize input update weights
        self.register_buffer('wzp', self._init_wzp())

        # Initialize update bias
        self.register_buffer('bz', self._init_bz())
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
        return Variable(torch.rand(self.pca_dim, self.hidden_dim))
    # end init_hidden

    # Init update-hidden matrix
    def init_wzh(self):
        """
        Init update-hidden matrix
        :return: Initiated update-hidden matrix
        """
        return Variable(torch.rand(self.pca_dim, self.hidden_dim))
    # end init_hidden

    # Reset learning
    def reset(self):
        """
        Reset learning
        :return:
        """
        # Reset output layer
        self.output.reset()

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
        # Compute hidden states
        hidden_states = self.esn_cell(u)

        # Resulting reduced stated
        pca_states = torch.zeros(1, hidden_states.size(1), self.pca_dim)

        # For each batch
        pca_states[0] = torch.from_numpy(self.ipca.fit_transform(hidden_states.data[0].numpy()).copy())
        pca_states = Variable(pca_states)

        # FFNN output
        return F.relu(self.linear2(F.relu(self.linear1(pca_states))))
    # end forward

    # Finish training
    def finalize(self):
        """
        Finalize training with LU factorization
        """
        # Finalize output training
        self.output.finalize()

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

# end GatedESN
