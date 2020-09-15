# -*- coding: utf-8 -*-
#
# File : echotorch/nn/reservoir/DeepESN.py
# Description : ESNs stacked on each others such as defined in Gallicchio, C., Micheli, A., & Pedrelli, L. (2017).
# Date : 10th of September, 2020
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
import numpy as np
import echotorch.utils.matrix_generation as mg
from echotorch.nn.linear.RRCell import RRCell
from ..Node import Node
from .LiESNCell import LiESNCell


# Deep ESN
class DeepESN(Node):
    """
    Deep ESN as defined in Gallicchio, C., Micheli, A., & Pedrelli, L. (2017).
    """

    # Constructor
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, w_generator, win_generator, wbias_generator,
                 leak_rate, input_scaling=1.0, nonlin_func=torch.tanh, learning_algo='inv', ridge_param=0.0,
                 with_bias=True, softmax_output=False, normalize_output=False, washout=0, create_rnn=True,
                 create_output=True, input_type='IF', output_type='AO', debug=Node.NO_DEBUG, test_case=None,
                 dtype=torch.float32):
        """
        Constructor
        :param n_layers: Number of layers to create
        :param input_dim: Input dimension size
        :param hidden_dim: Size of the reservoirs (all layer have the same size)
        :param output_dim: Output dimension size
        :param w_generator: Generator for the reservoir-to-reservoir matrices
        :param win_generator: Generator for the input-to-reservoir matrices (inputs for first layer, inputs for other from previous layer).
        :param wbias_generator: Generator for the internal biaises
        :param input_scaling: Input scaling (first layer)
        :param nonlin_func: Activation function (all layers)
        :param learning_algo: Learning algorithm (output layer) as 'inv' or 'pinv'
        :param ridge_param: Regularization parameter (output layer)
        :param with_bias: Add a bias to the output layer
        :param softmax_output: Add a softmax layer after the output layer ?
        :param washout: Washout period (ignore timesteps at the beginning of each sample)
        :param create_rnn: Create the RNN layers ?
        :param create_output: Create the output layer ?
        :param input_type: Input variant (IF: input-to-first, IA: input-to-all, GE: grouped-ESNs)
        :param output_type: Output flavour (AO: all-to-outputs, LO: last-to-outputs)
        :param debug: Debug mode
        :param test_case: Test case to call for test
        :param dtype: Data type
        """
        super(DeepESN, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            debug=debug,
            test_case=test_case,
            dtype=dtype
        )

        # Properties
        self._n_layers = n_layers
        self._hidden_dim = hidden_dim
        self._with_bias = with_bias
        self._w_generator = w_generator
        self._win_generator = win_generator
        self._wbias_generator = wbias_generator
        self._input_type = input_type
        self._washout = washout
        self._dtype = dtype

        # List of reservoirs
        self._reservoirs = list()

        # Create each layer
        if create_rnn:
            for layer_i in range(self._n_layers):
                # Generate matrices
                w, w_in, w_bias = self._generate_matrices(
                    self._get_hyperparam_value(w_generator, layer_i),
                    self._get_hyperparam_value(win_generator, layer_i),
                    self._get_hyperparam_value(wbias_generator, layer_i),
                    layer_i == 0
                )

                # Input dim
                layer_input_dim = w_in.size(1)

                # Recurrent layer
                esn_cell = LiESNCell(
                    input_dim=layer_input_dim,
                    output_dim=hidden_dim,
                    w=w,
                    w_in=w_in,
                    w_bias=w_bias,
                    leaky_rate=self._get_hyperparam_value(leak_rate, layer_i),
                    input_scaling=self._get_hyperparam_value(input_scaling, layer_i),
                    nonlin_func=self._get_hyperparam_value(nonlin_func, layer_i),
                    washout=washout,
                    debug=debug,
                    test_case=test_case,
                    dtype=dtype
                )

                # Add
                self._reservoirs.append(esn_cell)
            # end for
        # end if

        # Output layer
        if create_output:
            self._output = RRCell(
                input_dim=hidden_dim * n_layers,
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

    # Append a layer
    def append_layer(self, esn_cell):
        """
        Append a layer
        :param esn_cell: The ESNCell object to append to the stack
        """
        self._reservoirs.append(esn_cell)
        self._n_layers += 1
    # end append_layer

    # region PRIVATE

    # Get hyperparameter value
    def _get_hyperparam_value(self, hyperparam, layer_i):
        """
        Get hyperparameter value
        :param hyperparam: Hyperparameter (a value or a list)
        :param layer_i: Which layer (integer)
        :return: Hyperparameter value for this layer
        """
        if type(hyperparam) == list or type(hyperparam) == np.ndarray or type(hyperparam) == torch.tensor:
            return hyperparam[layer_i]
        else:
            return hyperparam
        # end if
    # end _get_hyperparam_value

    # Generate matrices
    def _generate_matrices(self, w_generator, win_generator, wbias_generator, first_layer=False):
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

        # Input matrix size
        if first_layer:
            win_size = (self._hidden_dim, self._input_dim)
        else:
            if self._input_type == 'IF':
                win_size = (self._hidden_dim, self._hidden_dim)
            elif self._input_type == 'IA':
                win_size = (self._hidden_dim, self._hidden_dim + self._input_dim)
            elif self._input_type == 'GE':
                win_size = (self._hidden_dim, self._input_dim)
            else:
                raise Exception("Unknown value for input_type : {}".format(self._input_type))
            # end if
        # end if

        # Generate Win matrix
        if isinstance(win_generator, mg.MatrixGenerator):
            w_in = win_generator.generate(size=win_size, dtype=self._dtype)
        elif callable(win_generator):
            w_in = win_generator(size=win_size, dtype=self._dtype)
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

    # region OVERRIDE

    # Forward
    def forward(self, u, y=None, reset_state=True):
        """
        Forward function
        :param u: Input signal
        :param y: Target outputs (or None if prediction)
        :param reset_state: Reset hidden state to zero or keep old one ?
        :return: Output (eval) or hidden states (train)
        """
        # Sizes
        time_length = int(u.size(1))
        batch_sizes = int(u.size(0))

        # Keep hidden states
        hidden_states = torch.zeros(batch_sizes, time_length, self._hidden_dim * self._n_layers, dtype=self._dtype)

        # Input to first layer
        layer_input = u

        # Compute hidden states for each layer
        for layer_i in range(self._n_layers):
            # Feed ESN
            if self._input_type == 'IF':
                layer_hidden_states = self._reservoirs[layer_i](layer_input, reset_state=reset_state)
                hidden_states[:, :, layer_i*self._hidden_dim:(layer_i+1)*self._hidden_dim] = layer_hidden_states
                layer_input = layer_hidden_states
            elif self._input_type == 'IA':
                # Add inputs for upper layers
                if layer_i > 0:
                    layer_input_with_u = torch.cat((layer_input, u), dim=2)
                else:
                    layer_input_with_u = u
                # end if

                # Go through ESN cell
                layer_hidden_states = self._reservoirs[layer_i](layer_input_with_u, reset_state=reset_state)
                hidden_states[:, :, layer_i * self._hidden_dim:(layer_i + 1) * self._hidden_dim] = layer_hidden_states
                layer_input = layer_hidden_states
            elif self._input_type == 'GE':
                layer_hidden_states = self._reservoirs[layer_i](u, reset_state=reset_state)
                hidden_states[:, :, layer_i * self._hidden_dim:(layer_i + 1) * self._hidden_dim] = layer_hidden_states
            else:
                raise Exception("Unknown input type : {}".format(self._input_type))
            # end if
        # end for

        # Learning algo
        if not self.training:
            return self._output(hidden_states, None)
        else:
            return self._output(hidden_states, y[:, self._washout:])
        # end if
    # end forward

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
        for layer_i in range(self._n_layers):
            self._reservoirs[layer_i].reset_hidden()
        # end for
    # end reset_hidden

    # Get item (get layer)
    def __getitem__(self, item):
        """
        Get item (get layer)
        :param item: Item index
        :return: ESNCell at item-th layer
        """
        return self._reservoirs[item]
    # end __getitem__

    # Set item (set layer)
    def __setitem__(self, key, value):
        """
        Set item (set layer)
        :param key: Layer index
        :param value: ESNCell object
        """
        self._reservoirs[key] = value
    # end __setitem__

    # Extra-information
    def extra_repr(self):
        """
        Extra-information
        :return: String
        """
        s = super(DeepESN, self).extra_repr()
        s += ', layers=[\n'
        for layer_i in range(self._n_layers):
            s += '\t{_reservoirs[' + str(layer_i) + ']},\n'
        # end for
        s += ']'
        return s.format(**self.__dict__)
    # end extra_repr

    # endregion OVERRIDE

# end DeepESN