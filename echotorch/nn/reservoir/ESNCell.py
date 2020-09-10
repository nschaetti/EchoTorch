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
from echotorch.utils.visualisation import Observable
from ..Node import Node


# Echo State Network layer
# Basis cell for ESN.
class ESNCell(Node, Observable):
    """
    Echo State Network layer
    Basis cell for ESN
    """

    # Constructor
    def __init__(self, input_dim, output_dim, w, w_in, w_bias, input_scaling=1.0, nonlin_func=torch.tanh, washout=0,
                 noise_generator=None, debug=Node.NO_DEBUG, test_case=None, dtype=torch.float32):
        """
        Constructor
        :param input_dim: Input dimension
        :param output_dim: Reservoir size
        :param input_scaling: Input scaling
        :param w: Internal weight matrix W
        :param w_in: Input-internal weight matrix Win
        :param w_bias: Internal units bias vector Wbias
        :param nonlin_func: Non-linear function applied to the units
        :param washout: Period to ignore in training at the beginning
        :param noise_generator: Noise generator used to add noise to states before non-linearity
        :param debug: Debug mode
        :param test_case: Test case to call for test.
        :param dtype: Data type used for vectors/matrices.
        """
        # Superclass
        super(ESNCell, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            debug=debug,
            dtype=dtype,
            test_case=test_case
        )

        # Init. Observable super-class
        Observable.__init__(self)

        # Params
        self._input_scaling = input_scaling
        self._nonlin_func = nonlin_func
        self._washout = washout
        self._noise_generator = noise_generator
        self._dtype = dtype

        # Init hidden state
        self.register_buffer('hidden', self._init_hidden())

        # Initialize input weights
        self.register_buffer('w_in', Variable(w_in, requires_grad=False))

        # Initialize reservoir weights randomly
        self.register_buffer('w', Variable(w, requires_grad=False))

        # Initialize bias
        self.register_buffer('w_bias', Variable(w_bias, requires_grad=False))

        # Add observation point
        self.add_observation_point("w", unique=True)
        self.add_observation_point("w_in", unique=True)
        self.add_observation_point("w_bias", unique=True)
        self.add_observation_point("X", unique=False)
        self.add_observation_point("U", unique=False)
    # end __init__

    # region PROPERTIES

    # Get washout
    @property
    def washout(self):
        """
        Get washout
        :return: Washout length
        """
        return self._washout
    # end washout

    # Set washout
    @washout.setter
    def washout(self, washout):
        """
        Washout
        :param washout: New washout
        """
        self._washout = washout
    # end washout

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
    # end spectral_radius

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

    # endregion PROPERTIES

    # region PUBLIC

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

        # For each sample
        for b in range(n_batches):
            # Reset hidden layer
            if reset_state:
                self.reset_hidden()
            # end if

            # Pre-update hook
            u[b, :] = self._pre_update_hook(u[b, :], self._forward_calls, b)

            # Observe inputs
            self.observation_point('U', u[b, :])

            # For each steps
            for t in range(time_length):
                # Current input
                ut = u[b, t] * self._input_scaling

                # Pre-hook
                ut = self._pre_step_update_hook(ut, self._forward_calls, b, t)

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
                x = self._post_step_update_hook(x.view(self.output_dim), ut, self._forward_calls, b, t)

                # Neural filter
                for neural_filter_handler in self._neural_filter_handlers:
                    x = neural_filter_handler(x, ut, self._forward_calls, b, t, t < self._washout)
                # end if

                # New last state
                self.hidden.data = x.data

                # Add to outputs
                outputs[b, t] = self.hidden
            # end for

            # Post-update hook
            outputs[b, :] = self._post_update_hook(outputs[b, :], u[b, :], self._forward_calls, b)

            # Post states update handlers
            for handler in self._post_states_update_handlers:
                handler(outputs[b, self._washout:], u[b, self._washout:], self._forward_calls, b)
            # end for

            # Observe states
            self.observation_point('X', outputs[b, self._washout:])
        # end for

        # Count calls to forward
        self._forward_calls += 1

        return outputs[:, self._washout:]
    # end forward

    # endregion PUBLIC

    # region PRIVATE

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
        :return: States before non-linearity
        """
        if self._noise_generator is None:
            return u_win + x_w + self.w_bias
        else:
            return u_win + x_w + self.w_bias + self._noise_generator(self._output_dim)
        # end if
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

    # endregion PRIVATE

    # region OVERRIDE

    # Extra-information
    def extra_repr(self):
        """
        Extra-information
        :return: String
        """
        s = super(ESNCell, self).extra_repr()
        s += ', nonlin_func={_nonlin_func}, washout={_washout}'
        return s.format(**self.__dict__)
    # end extra_repr

    # endregion OVERRIDE

# end ESNCell
