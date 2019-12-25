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
from echotorch.nn.linear.RRCell import RRCell
from .SPESNCell import SPESNCell
from ..reservoir import ESN
from ..Node import Node


# Self-Predicting Echo State Network module.
class SPESN(ESN):
    """
    Echo State Network module.
    """
    # region BODY

    # Constructor
    def __init__(self, input_dim, hidden_dim, output_dim, w_generator, win_generator, wbias_generator,
                 input_scaling=1.0, nonlin_func=torch.tanh, learning_algo='inv',
                 w_learning_algo='inv', ridge_param=0.000001, w_ridge_param=0.0001, with_bias=True,
                 softmax_output=False, washout=0, fill_left=False, loading_method=SPESNCell.W_LOADING,
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
        :param washout: Washout period (ignore timesteps at the beginning of each sample)
        :param debug: Debug mode
        :param test_case: Test case to call
        :param dtype: Data type
        """
        super(SPESN, self).__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            w_generator=w_generator,
            win_generator=win_generator,
            wbias_generator=wbias_generator,
            ridge_param=ridge_param,
            create_rnn=False,
            create_output=False,
            washout=washout,
            debug=debug,
            test_case=test_case,
            dtype=dtype
        )

        # Properties
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._with_bias = with_bias
        self._washout = washout
        self._w_generator = w_generator
        self._win_generator = win_generator
        self._wbias_generator = wbias_generator
        self._dtype = dtype

        # Generate matrices
        w, w_in, w_bias = self._generate_matrices(w_generator, win_generator, wbias_generator)

        # Recurrent layer
        self._esn_cell = SPESNCell(
            input_dim=input_dim,
            output_dim=hidden_dim,
            w=w,
            w_in=w_in,
            w_bias=w_bias,
            input_scaling=input_scaling,
            nonlin_func=nonlin_func,
            w_learning_algo=w_learning_algo,
            w_ridge_param=w_ridge_param,
            washout=washout,
            fill_left=fill_left,
            loading_method=loading_method,
            debug=debug,
            test_case=test_case,
            dtype=dtype
        )

        # Output layer
        self._output = RRCell(
            input_dim=hidden_dim,
            output_dim=output_dim,
            ridge_param=ridge_param,
            with_bias=with_bias,
            learning_algo=learning_algo,
            softmax_output=softmax_output,
            debug=debug,
            test_case=test_case,
            dtype=dtype
        )

        # Trainable elements
        self.add_trainable(self._esn_cell)
        self.add_trainable(self._output)
    # end __init__

    # region PROPERTIES
    # endregion PROPERTIES

    # region PUBLIC
    # endregion PUBLIC

    # region PRIVATE
    # endregion PRIVATE

    # endregion BODY
# end SPESN
