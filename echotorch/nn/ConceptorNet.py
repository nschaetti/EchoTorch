# -*- coding: utf-8 -*-
#
# File : echotorch/nn/ConceptorNet.py
# Description : A ESN-based Conceptor Network.
# Date : 17th of January, 2019
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
Created on 17 January 2019
@author: Nils Schaetti
"""

# Imports
import torch
import torch.nn as nn
from .ConceptorNetCell import ConceptorNetCell
from .RRCell import RRCell


# ESN-based ConceptorNet
class ConceptorNet(nn.Module):
    """
    ESN-based ConceptorNet
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0,
                 w=None, w_in=None, w_bias=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None,
                 leaky_rate=1.0, nonlin_func=torch.tanh, learning_algo='inv', ridge_param=0.0,
                 with_bias=True):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param hidden_dim: Hidden layer dimension
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
        super(ConceptorNet, self).__init__()

        # Properties
        self.with_bias = with_bias

        # Recurrent layer
        self.esn_cell = ConceptorNetCell(leaky_rate, False, input_dim, hidden_dim, spectral_radius=spectral_radius,
                                         bias_scaling=bias_scaling, input_scaling=input_scaling,
                                         w=w, w_in=w_in, w_bias=w_bias, sparsity=sparsity, input_set=input_set,
                                         w_sparsity=w_sparsity, nonlin_func=nonlin_func, feedbacks=False,
                                         feedbacks_dim=input_dim, wfdb_sparsity=None,
                                         normalize_feedbacks=False)
        # end if

        # Input recreation weights layer (Ridge regression)
        self.input_recreation = RRCell(hidden_dim, input_dim, ridge_param, None, with_bias, learning_algo)
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

    # Input recreation matrix
    def input_recreation_matrix(self):
        """
        Input recreation matrix
        :return:
        """
        return self.input_recreation.get_w_out()
    # end input_recreation_matrix

    ###############################################
    # PUBLIC
    ###############################################

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

    # Output matrix
    def get_w_out(self):
        """
        Output matrix
        :return:
        """
        return self.output.w_out
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
    def forward(self, u, c, reset_state=True):
        """
        Forward
        :param u: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        # Compute hidden states
        if self.training:
            hidden_states = self.esn_cell(
                u,
                reset_state=reset_state
            )

            # Learning input recreation
            self.input_recreation(hidden_states, u)

            # Learning conceptor
            return c(hidden_states, hidden_states)
        else:
            hidden_states = self.esn_cell(
                u=None,
                reset_state=reset_state,
                input_recreation=self.input_recreation,
                conceptor=c
            )

            # Return observed outputs
            return self.input_recreation(hidden_states)
        # end if
    # end forward

    # Finish training
    def finalize(self):
        """
        Finalize training with LU factorization
        """
        # Finalize input recreation training
        self.input_recreation.finalize()

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

# end ESNCell
