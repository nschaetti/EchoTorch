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
from EchoTorch.nn.ESNCell import ESNCell


# Echo State Network module
class ESN(nn.Module):
    """
    Echo State Network module
    """

    # Constructor
    def __init__(self, input_dim, output_dim, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0, w=None, w_in=None,
                 w_bias=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None,
                 nonlin_func=torch.tanh):
        """
        Constructor
        :param input_dim: Inputs dimension.
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
        """
        super(ESN, self).__init__()

        # Layers
        self.esn_cell = ESNCell(input_dim, output_dim, spectral_radius, bias_scaling, input_scaling, w, w_in, w_bias,
                                sparsity, input_set, w_sparsity, nonlin_func)
        self.linear = nn.Linear(output_dim, 1, bias=True)
    # end __init__

    ###############################################
    # PUBLIC
    ###############################################

    # Forward
    def forward(self, u, hidden):
        """
        Forward
        :param u: Input signal.
        :param hidden: Hidden layer state (x).
        :return: Resulting hidden states.
        """
        # Get hidden states
        hidden_states = self.esn_cell(u, hidden)

        # Linear output
        out = self.linear(hidden_states)

        return out
    # end forward

    # Init hidden layer
    def init_hidden(self):
        """
        Init hidden layer.
        :return: Initiated hidden layer
        """
        return self.esn_cell.init_hidden()
    # end init_hidden

    # Get W's spectral radius
    def get_spectral_radius(self):
        """
        Get W's spectral radius.
        :return: W's spectral radius
        """
        return self.esn_cell.get_spectral_raduis()
    # end spectral_radius

    ###############################################
    # PRIVATE
    ###############################################

# end ESNCell
