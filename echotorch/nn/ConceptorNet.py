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
import matplotlib.pyplot as plt


# ESN-based ConceptorNet
class ConceptorNet(nn.Module):
    """
    ESN-based ConceptorNet
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, output_dim=None, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0,
                 w=None, w_in=None, w_bias=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None,
                 leaky_rate=1.0, nonlin_func=torch.tanh, learning_algo='inv', ridge_param=0.0,
                 with_bias=True, seed=None, washout=1, w_distrib='uniform', win_distrib='uniform',
                 wbias_distrib='uniform', win_normal=(0.0, 1.0), w_normal=(0.0, 1.0), wbias_normal=(0.0, 1.0),
                 w_ridge_param=0.0, dtype=torch.float32):
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
        self.washout = washout
        self.hidden_dim = hidden_dim

        # Recurrent layer
        self.esn_cell = ConceptorNetCell(leaky_rate, False, input_dim, hidden_dim, spectral_radius=spectral_radius,
                                         bias_scaling=bias_scaling, input_scaling=input_scaling,
                                         w=w, w_in=w_in, w_bias=w_bias, sparsity=sparsity, input_set=input_set,
                                         w_sparsity=w_sparsity, nonlin_func=nonlin_func, feedbacks=False,
                                         feedbacks_dim=input_dim, wfdb_sparsity=None,
                                         normalize_feedbacks=False, seed=seed, w_distrib=w_distrib,
                                         win_distrib=win_distrib, wbias_distrib=wbias_distrib, win_normal=win_normal,
                                         w_normal=w_normal, wbias_normal=wbias_normal, dtype=dtype)
        # end if

        # Input recreation weights layer (Ridge regression)
        self.input_recreation = RRCell(
            hidden_dim,
            hidden_dim,
            w_ridge_param,
            None,
            with_bias=False,
            learning_algo=learning_algo,
            softmax_output=False,
            averaged=True,
            dtype=dtype
        )

        # Output (state observer)
        if output_dim is not None:
            self.output = RRCell(
                hidden_dim,
                output_dim,
                ridge_param,
                None,
                with_bias=False,
                learning_algo=learning_algo,
                softmax_output=False,
                averaged=False,
                dtype=dtype
            )
        else:
            self.output = None
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

    # Input recreation matrix
    @property
    def input_recreation_matrix(self):
        """
        Input recreation matrix
        :return:
        """
        return self.input_recreation.get_w_out()
    # end input_recreation_matrix

    ###############################################
    # PRIVATE
    ###############################################

    # Arctanh
    def arctanh(self, x):
        """
        Inverse tanh
        :param x:
        :return:
        """
        return 0.5 * torch.log((1 + x) / (1 - x))
    # end arctanh

    ###############################################
    # PUBLIC
    ###############################################

    # Mode
    def set_train(self):
        """
        Mode
        :return:
        """
        self.train(True)
        self.input_recreation.train(True)
        if self.output is not None:
            self.output.train(True)
        # end if
    # end set_train

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
    def forward(self, u=None, y=None, c=None, reset_state=True, length=None, mu=None, return_states=False):
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

            # Batch size and time length
            batch_size = hidden_states.shape[0]

            # X and Washout
            x = hidden_states[:, self.washout:]
            time_length = x.shape[1]

            # Past hidden states
            x_tilda = hidden_states[:, self.washout-1:-1]

            # Bias
            bias = self.esn_cell.w_bias[0].expand(batch_size, time_length, self.hidden_dim)

            # Learning input recreation
            self.input_recreation(x_tilda, self.arctanh(x) - bias)

            # Learning state observer
            if self.output is not None and y is not None:
                self.output(x, y[:, self.washout:])
            # end if

            # Learning conceptor
            return c(x, x)
        elif c is None:
            hidden_states = self.esn_cell(
                u,
                reset_state=reset_state,
            )

            # Return outputs or states
            if self.output is not None and not return_states:
                return self.output(hidden_states)
            else:
                return hidden_states
            # end if
        else:
            hidden_states = self.esn_cell(
                u=u,
                reset_state=reset_state,
                input_recreation=self.input_recreation,
                conceptor=c,
                length=length,
                mu=mu
            )

            # Return outputs of states
            if self.output is not None:
                return self.output(hidden_states)
            else:
                return self.input_recreation(hidden_states)
            # end if
        # end if
    # end forward

    # Finish training
    def finalize(self, train=False):
        """
        Finalize training with LU factorization
        """
        # Finalize input recreation training
        self.input_recreation.finalize()

        # Finalize output
        if self.output is not None:
            self.output.finalize()
        # end if

        # Not in training mode anymore
        self.train(train)
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
