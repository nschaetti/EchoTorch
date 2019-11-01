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

# Imports
import torch
import torch.sparse
from torch.autograd import Variable
import echotorch.utils
from ..Node import Node


# Echo State Network layer
# Basis cell for ESN.
class ESNCell(Node):
    """
    Echo State Network layer
    Basis cell for ESN
    """

    # Constructor
    def __init__(self, input_dim, output_dim, w, w_in, w_bias, spectral_radius=0.9, bias_scaling=0,
                 input_scaling=1.0, nonlin_func=torch.tanh, washout=0, dtype=torch.float32):
        """
        Constructor
        :param input_dim: Input dimension
        :param output_dim: Reservoir size
        :param spectral_radius: Spectral radius
        :param bias_scaling: Bias scaling
        :param input_scaling: Input scaling
        :param w: Internal weight matrix W
        :param w_in: Input-internal weight matrix Win
        :param w_bias: Internal units bias vector Wbias
        :param nonlin_func: Non-linear function applied to the units
        :param washout: Period to ignore in training at the beginning
        :param dtype: Data type used for vectors/matrices.
        """
        # Superclass
        super(ESNCell, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            dtype=dtype
        )

        # Params
        self._spectral_radius = spectral_radius
        self._bias_scaling = bias_scaling
        self._input_scaling = input_scaling
        self._nonlin_func = nonlin_func
        self._washout = washout
        self._dtype = dtype

        # Init hidden state
        self.register_buffer('hidden', self._init_hidden())

        # Initialize input weights
        self.register_buffer('w_in', self._scale_win(w_in))

        # Initialize reservoir weights randomly
        self.register_buffer('w', self._scale_w(w))

        # Initialize bias
        self.register_buffer('w_bias', self._scale_wbias(w_bias))
    # end __init__

    ######################
    # PROPERTIES
    ######################

    # ESN cell
    @property
    def cell(self):
        """
        ESN cell
        :return: ESN cell
        """
        return self._esn_cell
    # end cell

    # Get W's spectral radius
    @property
    def spectral_radius(self):
        """
        Get W's spectral radius
        :return: W's spectral radius
        """
        return echotorch.utils.spectral_radius(self.w)
    # end spectral_radius

    # Change spectral radius
    @spectral_radius.setter
    def spectral_radius(self, sp):
        """
        Change spectral radius
        :param sp: New spectral radius
        """
        self.w *= sp / echotorch.utils.spectral_radius(self.w)
        self._spectral_radius = sp
    # end spectral_radius

    # Get bias scaling
    @property
    def bias_scaling(self):
        """
        Get bias scaling
        :return: Bias scaling parameter
        """
        return self._bias_scaling
    # end bias_scaling

    # Get input scaling
    @property
    def input_scaling(self):
        """
        Get input scaling
        :return: Input scaling parameters
        """
        return self._input_scaling
    # end input_scaling

    # Get non linear function
    @property
    def nonlin_func(self):
        """
        Get non linear function
        :return: Non linear function
        """
        return self._nonlin_func
    # end nonlin_func

    ######################
    # PUBLIC
    ######################

    # Reset hidden layer
    def reset_hidden(self):
        """
        Reset hidden layer
        :return:
        """
        self.hidden.fill_(0.0)
    # end reset_hidden

    # Set hidden layer
    def set_hidden(self, x):
        """
        Set hidden layer
        :param x:
        :return:
        """
        self.hidden.data = x.data
    # end set_hidden

    # Forward
    def forward(self, u, reset_state=True):
        """
        Forward pass function
        :param u: Input signal
        :param reset_state: Reset state at each batch ?
        :return: Resulting hidden states
        """
        # Time length
        time_length = int(u.size()[1])

        # Number of batches
        n_batches = int(u.size()[0])

        # Outputs
        outputs = Variable(torch.zeros(n_batches, time_length, self.output_dim, dtype=self.dtype))
        outputs = outputs.cuda() if self.hidden.is_cuda else outputs

        # For each batch
        for b in range(n_batches):
            # Reset hidden layer
            if reset_state:
                self.reset_hidden()
            # end if

            # Pre-update hook
            u[b, :] = self._pre_update_hook(u[b, :], b)

            # For each steps
            for t in range(time_length):
                # Current input
                ut = u[b, t]

                # Pre-hook
                ut = self._pre_step_update_hook(ut, t)

                # Compute input layer
                u_win = self._input_layer(ut)

                # Apply W to x
                x_w = self._recurrent_layer(self.hidden)

                # Add everything
                x = self._reservoir_layer(u_win, x_w)

                # Apply activation function
                x = self.nonlin_func(x)

                # Post nonlinearity
                x = self._post_nonlinearity(x)

                # Post-hook
                x = self._post_step_update_hook(x.view(self.output_dim), ut, t)

                # Neural filter
                if self._neural_filter_handler is not None:
                    x = self._neural_filter_handler(x, ut, t, t < self._washout)
                # end if

                # New last state
                self.hidden.data = x.data

                # Add to outputs
                outputs[b, t] = self.hidden
            # end for

            # Post-update hook
            outputs[b, :] = self._post_update_hook(outputs[b, :], u[b, :], b)
        # end for

        return outputs[:, self._washout:]
    # end forward

    ######################
    # PRIVATE
    ######################

    # Compute post nonlinearity hook
    def _post_nonlinearity(self, x):
        """
        Compute post nonlinearity hook
        :param x: Reservoir state at time t
        :return: Reservoir state
        """
        return x
    # end _post_nonlinearity

    # Compute reservoir layer
    def _reservoir_layer(self, u_win, x_w):
        """
        Compute reservoir layer
        :param u_win: Processed inputs
        :param x_w: Processed states
        :return: States before nonlinearity
        """
        return u_win + x_w + self.w_bias
    # end _reservoir_layer

    # Compute recurrent layer
    def _recurrent_layer(self, xt):
        """
        Compute recurrent layer
        :param xt: Reservoir state at t-1
        :return: Processed state
        """
        return self.w.mv(self.hidden)
    # end _recurrent_layer

    # Compute input layer
    def _input_layer(self, ut):
        """
        Compute input layer
        :param ut: Inputs
        :return: Processed inputs
        """
        return self.w_in.mv(ut)
    # end _input_layer

    # Init hidden layer
    def _init_hidden(self):
        """
        Init hidden layer
        :return: Initiated hidden layer
        """
        return Variable(torch.zeros(self.output_dim, dtype=self.dtype), requires_grad=False)
    # end _init_hidden

    # Scale W matrix
    def _scale_w(self, w):
        """
        Scale W matrix
        :return: Scaled W matrix
        """
        # Scale it to spectral radius
        # w *= self._spectral_radius / echotorch.utils.spectral_radius(w)
        w *= self._spectral_radius
        return Variable(w, requires_grad=False)
    # end _scale_w

    # Scale Win matrix
    def _scale_win(self, w_in):
        """
        Scale Win matrix
        :return: Scaled Win
        """
        # Initialize input weight matrix
        w_in *= self._input_scaling
        return Variable(w_in, requires_grad=False)
    # end _scale_win

    # Scale Wbias matrix
    def _scale_wbias(self, w_bias):
        """
        Scale Wbias matrix
        :return: Scaled bias matrix
        """
        # Initialize bias matrix
        w_bias *= self._bias_scaling
        return Variable(w_bias, requires_grad=False)
    # end _scale_wbias

# end ESNCell
