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
import echotorch.utils.matrix_generation as mg
from echotorch.nn.linear.RRCell import RRCell
from ..Node import Node
from .ESNCell import ESNCell


# Echo State Network module.
class ESN(Node):
    """
    Echo State Network module.
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, output_dim, w_generator, win_generator, wbias_generator,
                 input_scaling=1.0, nonlin_func=torch.tanh, learning_algo='inv', ridge_param=0.0, with_bias=True,
                 softmax_output=False, normalize_output=False, washout=0, create_rnn=True, create_output=True,
                 debug=Node.NO_DEBUG, test_case=None, dtype=torch.float32):
        """
        Constructor
        :param input_dim: Input feature space dimension
        :param hidden_dim: Hidden space dimension
        :param output_dim: Output space dimension
        :param w_generator: Internal weight matrix generator
        :param win_generator: Input-output weight matrix generator
        :param wbias_generator: Bias matrix generator
        :param input_scaling: Input scaling
        :param nonlin_func: Non-linear function
        :param learning_algo: Learning method (inv, pinv)
        :param ridge_param: Ridge parameter
        :param with_bias: Add a bias to the output layer ?
        :param softmax_output: Add a softmax output layer
        :param normalize_output: Normalize output to sum to one
        :param washout: Washout period (ignore timesteps at the beginning of each sample)
        :param create_rnn: Create RNN layer ?
        :param create_output: Create the output layer ?
        :param debug: Debug mode
        :param test_case: Test case to call for test
        :param dtype: Data type
        """
        super(ESN, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            debug=debug,
            test_case=test_case,
            dtype=dtype
        )

        # Properties
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._with_bias = with_bias
        self._w_generator = w_generator
        self._win_generator = win_generator
        self._wbias_generator = wbias_generator
        self._dtype = dtype

        if create_rnn:
            # Generate matrices
            w, w_in, w_bias = self._generate_matrices(w_generator, win_generator, wbias_generator)

            # Recurrent layer
            self._esn_cell = ESNCell(
                input_dim=input_dim,
                output_dim=hidden_dim,
                w=w,
                w_in=w_in,
                w_bias=w_bias,
                input_scaling=input_scaling,
                nonlin_func=nonlin_func,
                washout=washout,
                debug=debug,
                test_case=test_case,
                dtype=dtype
            )
        # end if

        # Output layer
        if create_output:
            self._output = RRCell(
                input_dim=hidden_dim,
                output_dim=output_dim,
                ridge_param=ridge_param,
                with_bias=with_bias,
                learning_algo=learning_algo,
                softmax_output=softmax_output,
                normalize_output=normalize_output,
                debug=debug,
                test_case=test_case,
                dtype=dtype
            )
            self.add_trainable(self._output)
        # end if
    # end __init__

    # region PROPERTIES

    # Get washout
    @property
    def washout(self):
        """
        Get washout
        :return: Washout length
        """
        return self._esn_cell.washout
    # end washout

    # Set washout
    @washout.setter
    def washout(self, washout):
        """
        Washout
        :param washout: New washout
        """
        self._esn_cell.washout = washout
    # end washout

    # ESN cell
    @property
    def cell(self):
        """
        ESN cell
        :return: ESN cell
        """
        return self._esn_cell
    # end cell

    # Output
    @property
    def output(self):
        """
        Output layer
        :return: Output layer
        """
        return self._output
    # end output

    # Get W's spectral radius
    @property
    def spectral_radius(self):
        """
        Get W's spectral radius
        :return: W's spectral radius
        """
        return self._esn_cell.spectral_radius
    # end spectral_radius

    # Change spectral radius
    @spectral_radius.setter
    def spectral_radius(self, sp):
        """
        Change spectral radius
        :param sp: New spectral radius
        """
        self._esn_cell.spectral_radius = sp
    # end spectral_radius

    # Get input scaling
    @property
    def input_scaling(self):
        """
        Get input scaling
        :return: Input scaling parameters
        """
        return self._esn_cell.input_scaling
    # end input_scaling

    # Get non linear function
    @property
    def nonlin_func(self):
        """
        Get non linear function
        :return: Non linear function
        """
        return self._esn_cell.nonlin_func
    # end nonlin_func

    # Hidden layer
    @property
    def hidden(self):
        """
        Hidden layer
        :return: Hidden layer
        """
        return self._esn_cell.hidden
    # end hidden

    # Hidden weight matrix
    @property
    def w(self):
        """
        Hidden weight matrix
        :return: Internal weight matrix
        """
        return self._esn_cell.w
    # end w

    # Input matrix
    @property
    def w_in(self):
        """
        Input matrix
        :return: Input matrix
        """
        return self._esn_cell.w_in
    # end w_in

    # Output matrix
    @property
    def w_out(self):
        """
        Output matrix
        :return: Output matrix
        """
        return self._output.w_out
    # end w_out

    # endregion PROPRETIES

    # region OVERRIDE

    # Forward
    def forward(self, u, y=None, reset_state=True):
        """
        Forward
        :param u: Input signal.
        :param y: Target outputs (or None if prediction)
        :return: Output or hidden states
        """
        # Compute hidden states
        hidden_states = self._esn_cell(u, reset_state=reset_state)

        # Learning algo
        if not self.training:
            return self._output(hidden_states, None)
        else:
            return self._output(hidden_states, y[:, self._esn_cell.washout:])
        # end if
    # end forward

    # region PUBLIC

    # Reset layer (not trained)
    def reset(self):
        """
        Reset layer (not trained)
        """
        # Reset output layer
        self._output.reset()

        # Training mode again
        self.train(True)
    # end reset

    # Reset hidden layer
    def reset_hidden(self):
        """
        Reset hidden layer
        :return:
        """
        self._esn_cell.reset_hidden()
    # end reset_hidden

    # endregion OVERRIDE

    # region PRIVATE

    # Generate matrices
    def _generate_matrices(self, w_generator, win_generator, wbias_generator):
        """
        Generate matrices
        :param w_generator: W matrix generator
        :param win_generator: Win matrix generator
        :param wbias_generator: Wbias matrix generator
        :return: W, Win, Wbias
        """
        # Generate W matrix
        if isinstance(w_generator, mg.MatrixGenerator):
            w = w_generator.generate(size=(self._hidden_dim, self._hidden_dim), dtype=self._dtype)
        elif callable(w_generator):
            w = w_generator(size=(self._hidden_dim, self._hidden_dim), dtype=self._dtype)
        else:
            w = w_generator
        # end if

        # Generate Win matrix
        if isinstance(win_generator, mg.MatrixGenerator):
            w_in = win_generator.generate(size=(self._hidden_dim, self._input_dim), dtype=self._dtype)
        elif callable(win_generator):
            w_in = win_generator(size=(self._hidden_dim, self._input_dim), dtype=self._dtype)
        else:
            w_in = win_generator
        # end if

        # Generate Wbias matrix
        if isinstance(wbias_generator, mg.MatrixGenerator):
            w_bias = wbias_generator.generate(size=self._hidden_dim, dtype=self._dtype)
        elif callable(wbias_generator):
            w_bias = wbias_generator(size=self._hidden_dim, dtype=self._dtype)
        else:
            w_bias = wbias_generator
        # end if

        return w, w_in, w_bias
    # end _generate_matrices

    # endregion PRIVATE

# end ESN
