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
from . import ESNCell
from .RRCell import RRCell


# Echo State Network module
class ESN(nn.Module):
    """
    Echo State Network module
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, output_dim, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0,
                 w=None, w_in=None, w_bias=None, w_fdb=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None,
                 nonlin_func=torch.tanh, learning_algo='inv', ridge_param=0.0, create_cell=True,
                 feedbacks=False, with_bias=True, wfdb_sparsity=None, normalize_feedbacks=False,
                 softmax_output=False, seed=None, washout=0, w_distrib='uniform', win_distrib='uniform',
                 wbias_distrib='uniform', win_normal=(0.0, 1.0), w_normal=(0.0, 1.0), wbias_normal=(0.0, 1.0),
                 dtype=torch.float32):
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
        self.feedbacks = feedbacks
        self.with_bias = with_bias
        self.normalize_feedbacks = normalize_feedbacks
        self.washout = washout
        self.dtype = dtype

        # Recurrent layer
        if create_cell:
            self.esn_cell = ESNCell(input_dim, hidden_dim, spectral_radius, bias_scaling, input_scaling, w, w_in,
                                    w_bias, w_fdb, sparsity, input_set, w_sparsity, nonlin_func, feedbacks, output_dim,
                                    wfdb_sparsity, normalize_feedbacks, seed, w_distrib, win_distrib, wbias_distrib,
                                    win_normal, w_normal, wbias_normal, dtype)
        # end if

        # Ouput layer
        self.output = RRCell(hidden_dim, output_dim, ridge_param, feedbacks, with_bias, learning_algo, softmax_output, dtype)
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
    def forward(self, u, y=None, reset_state=True):
        """
        Forward
        :param u: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        # Compute hidden states
        if self.feedbacks and self.training:
            hidden_states = self.esn_cell(u, y, reset_state=reset_state)
        elif self.feedbacks and not self.training:
            hidden_states = self.esn_cell(u, w_out=self.output.w_out, reset_state=reset_state)
        else:
            hidden_states = self.esn_cell(u, reset_state=reset_state)
        # end if

        # Learning algo
        if y is not None:
            return self.output(hidden_states[:, self.washout:], y[:, self.washout:])
        else:
            return self.output(hidden_states[:, self.washout:], y)
        # end if
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

    # Get W's spectral radius
    def get_spectral_radius(self):
        """
        Get W's spectral radius
        :return: W's spectral radius
        """
        return self.esn_cell.get_spectral_raduis()
    # end spectral_radius

# end ESNCell
