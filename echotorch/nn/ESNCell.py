# -*- coding: utf-8 -*-
#
# File : echotorch/nn/ESNCell.py
# Description : An Echo State Network layer.
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
from torch.autograd import Variable
import torch.nn as nn
import EchoTorch.tools
import numpy as np


# Echo State Network layer
class ESNCell(nn.Module):
    """
    Echo State Network layer
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
        super(ESNCell, self).__init__()

        # Params
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.bias_scaling = bias_scaling
        self.input_scaling = input_scaling
        self.w = w
        self.w_in = w_in
        self.w_bias = w_bias
        self.sparsity = sparsity
        self.input_set = input_set
        self.w_sparsity = w_sparsity
        self.nonlin_func = nonlin_func

        # Initialize inout weights
        self.w_in = self._generate_win()

        # Initialize reservoir weights randomly
        self.w = self._generate_w()

        # Initialize bias
        self.w_bias = self._generate_wbias()
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
        # Steps
        steps = int(u.size()[0])

        # Outputs
        outputs = Variable(torch.zeros(steps, self.output_dim))

        # For each steps
        for i in range(steps):
            # Current input
            ut = u[i]

            # Compute input layer
            u_win = self.w_in.mv(ut)

            # Apply W to x
            x_w = self.w.mv(hidden)

            # Apply activation function
            x_w = self.nonlin_func(x_w)

            # Add everything
            x = u_win + x_w + self.w_bias

            # Add to outputs
            hidden = x.view(self.output_dim)

            # New last state
            outputs[i] = hidden
        # end for

        return outputs
    # end forward

    # Init hidden layer
    def init_hidden(self):
        """
        Init hidden layer
        :return: Initiated hidden layer
        """
        return Variable(torch.zeros(self.output_dim))
    # end init_hidden

    # Get W's spectral radius
    def get_spectral_radius(self):
        """
        Get W's spectral radius
        :return: W's spectral radius
        """
        return EchoTorch.tools.spectral_radius(self.w)
    # end spectral_radius

    ###############################################
    # PRIVATE
    ###############################################

    # Generate W matrix
    def _generate_w(self):
        """
        Generate W matrix
        :return: The generated W matrix as Variable
        """
        # Initialize reservoir weight matrix
        if self.w is None:
            # Sparsity
            if self.w_sparsity is None:
                w = torch.rand(self.output_dim, self.output_dim) * 2.0 - 1.0
            else:
                w = np.random.choice([0.0, 1.0], (self.output_dim, self.output_dim), p=[1.0-self.w_sparsity, self.w_sparsity])
                w[w == 1] = np.random.rand(len(w[w == 1])) * 2.0 - 1.0
                w = torch.from_numpy(w.astype(np.float32))
            # end if
        else:
            if callable(self.w):
                w = self.w(self.output_dim)
            else:
                w = self.w
            # end if
        # end if

        # Scale it to spectral radius
        w *= self.spectral_radius / EchoTorch.tools.spectral_radius(w)

        return Variable(w, requires_grad=False)
    # end generate_W

    # Generate Win matrix
    def _generate_win(self):
        """
        Generate Win matrix
        :return: The generated Win matrix as Variable
        """
        # Initialize input weight matrix
        if self.w_in is None:
            if self.sparsity is None:
                w_in = self.input_scaling * (
                            np.random.randint(0, 2, (self.output_dim, self.input_dim)) * 2.0 - 1.0)
                w_in = torch.from_numpy(w_in.astype(np.float32))
            else:
                w_in = self.input_scaling * np.random.choice(np.append([0], self.input_set),
                                                             (self.output_dim, self.input_dim),
                                                             p=np.append([1.0 - self.sparsity],
                                                                         [self.sparsity / len(self.input_set)] * len(
                                                                             self.input_set)))
                w_in = torch.from_numpy(w_in.astype(np.float32))
            # end if
        else:
            if callable(self.w_in):
                w_in = self.w_in(self.output_dim, self.input_dim)
            else:
                w_in = self.w_in
            # end if
        # end if

        return Variable(w_in, requires_grad=False)
    # end _generate_win

    # Generate Wbias matrix
    def _generate_wbias(self):
        """
        Generate Wbias matrix
        :return: The generated Wbias matrix as Variable
        """
        # Initialize bias matrix
        if self.w_bias is None:
            w_bias = self.bias_scaling * (torch.rand(1, self.output_dim) * 2.0 - 1.0)
        else:
            if callable((self.w_bias)):
                w_bias = self.w_bias(self.output_dim)
            else:
                w_bias = self.w_bias
            # end if
        # end if

        return Variable(w_bias, requires_grad=False)
    # end _generate_wbias

# end ESNCell
