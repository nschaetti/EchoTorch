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
import torch.sparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from .LiESNCell import LiESNCell
from ..linear.RRCell import RRCell
from .ESNCell import ESNCell
import numpy as np


# Stacked Echo State Network module
class StackedESN(nn.Module):
    """
    Stacked Echo State Network module
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, output_dim, leaky_rate=1.0, spectral_radius=0.9, bias_scaling=0,
                 input_scaling=1.0, w=None, w_in=None, w_bias=None, sparsity=None, input_set=(1.0, -1.0),
                 w_sparsity=None, nonlin_func=torch.tanh, learning_algo='inv', ridge_param=0.0, with_bias=True):
        """
        Constructor

        Arguments:
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
        super(StackedESN, self).__init__()

        # Properties
        self.n_layers = len(hidden_dim)
        self.esn_layers = list()

        # Number of features
        self.n_features = 0

        # Recurrent layer
        for n in range(self.n_layers):
            # Input dim
            layer_input_dim = input_dim if n == 0 else hidden_dim[n-1]

            # Final state size
            self.n_features += hidden_dim[n]

            # Parameters
            layer_leaky_rate = leaky_rate[n] if type(leaky_rate) is list or type(leaky_rate) is np.ndarray else leaky_rate
            layer_spectral_radius = spectral_radius[n] if type(spectral_radius) is list or type(spectral_radius) is np.ndarray else spectral_radius
            layer_bias_scaling = bias_scaling[n] if type(bias_scaling) is list or type(bias_scaling) is np.ndarray else bias_scaling
            layer_input_scaling = input_scaling[n] if type(input_scaling) is list or type(input_scaling) is np.ndarray else input_scaling

            # W
            if type(w) is torch.Tensor and w.dim() == 3:
                layer_w = w[n]
            elif type(w) is torch.Tensor:
                layer_w = w
            else:
                layer_w = None
            # end if

            # W in
            if type(w_in) is torch.Tensor and w_in.dim() == 3:
                layer_w_in = w_in[n]
            elif type(w_in) is torch.Tensor:
                layer_w_in = w_in
            else:
                layer_w_in = None
            # end if

            # W bias
            if type(w_bias) is torch.Tensor and w_bias.dim() == 2:
                layer_w_bias = w_bias[n]
            elif type(w_bias) is torch.Tensor:
                layer_w_bias = w_bias
            else:
                layer_w_bias = None
            # end if

            # Parameters
            layer_sparsity = sparsity[n] if type(sparsity) is list or type(sparsity) is np.ndarray else sparsity
            layer_input_set = input_set[n] if type(input_set) is list or type(input_set) is np.ndarray else input_set
            layer_w_sparsity = w_sparsity[n] if type(w_sparsity) is list or type(w_sparsity) is np.ndarray else w_sparsity
            layer_nonlin_func = nonlin_func[n] if type(nonlin_func) is list or type(nonlin_func) is np.ndarray else nonlin_func

            # Create LiESN cell
            self.esn_layers.append(LiESNCell(
                layer_leaky_rate, False, layer_input_dim, hidden_dim[n], layer_spectral_radius, layer_bias_scaling,
                layer_input_scaling, layer_w, layer_w_in, layer_w_bias, None, layer_sparsity, layer_input_set,
                layer_w_sparsity, layer_nonlin_func
            ))
        # end for

        # Output layer
        self.output = RRCell(self.n_features, output_dim, ridge_param, False, with_bias, learning_algo)
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
        # Hidden states
        hidden_states = list()

        # For each ESN
        for esn_cell in self.esn_layers:
            hidden_states.append(esn_cell.hidden)
        # end for

        return hidden_states
    # end hidden

    # Hidden weight matrix
    @property
    def w(self):
        """
        Hidden weight matrix
        :return:
        """
        # W
        w_mtx = list()

        # For each ESN
        for esn_cell in self.esn_layers:
            w_mtx.append(esn_cell.w)
        # end for

        return w_mtx
    # end w

    # Input matrix
    @property
    def w_in(self):
        """
        Input matrix
        :return:
        """
        # W in
        win_mtx = list()

        # For each ESN
        for esn_cell in self.esn_layers:
            win_mtx.append(esn_cell.w_in)
        # end for

        return win_mtx
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

    # Forward
    def forward(self, u, y=None):
        """
        Forward
        :param u: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        # Hidden states
        hidden_states = Variable(torch.zeros(u.size(0), u.size(1), self.n_features))

        # Compute hidden states
        pos = 0
        for index, esn_cell in enumerate(self.esn_layers):
            layer_dim = esn_cell.output_dim
            if index == 0:
                last_hidden_states = esn_cell(u)
            else:
                last_hidden_states = esn_cell(last_hidden_states)
            # end if

            # Update
            hidden_states[:, :, pos:pos + layer_dim] = last_hidden_states

            # Next position
            pos += layer_dim
        # end for

        # Learning algo
        return self.output(hidden_states, y)
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

    ############################################
    # STATIC
    ############################################

    # Generate W matrices for a stacked ESN
    @staticmethod
    def generate_ws(n_layers, reservoir_size, w_sparsity):
        """
        Generate W matrices for a stacked ESN
        :param n_layers:
        :param reservoir_size:
        :param w_sparsity:
        :return:
        """
        ws = torch.FloatTensor(n_layers, reservoir_size, reservoir_size)
        for i in range(n_layers):
            ws[i] = ESNCell.generate_w(reservoir_size, w_sparsity)
        # end for
        return ws
    # end for

# end ESNCell
