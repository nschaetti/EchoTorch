# -*- coding: utf-8 -*-
#
# File : echotorch/nn/LiESN.py
# Description : An Leaky-Rate Echo State Network module.
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
from .LiESNCell import LiESNCell
from echotorch.nn.reservoir.ESN import ESN
from ..Node import Node


# Leaky-Integrated Echo State Network module
class LiESN(ESN):
    """
    Leaky-Integrated Echo State Network module
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, output_dim, leaky_rate, w_generator, win_generator, wbias_generator,
                 spectral_radius=0.9, bias_scaling=1.0, input_scaling=1.0, nonlin_func=torch.tanh, learning_algo='inv',
                 ridge_param=0.0, with_bias=True, softmax_output=False, washout=0, debug=Node.NO_DEBUG, test_case=None,
                 dtype=torch.float32):
        """
        Constructor
        :param input_dim: Input feature space dimension
        :param hidden_dim: Reservoir hidden space dimension
        :param output_dim: Output space dimension
        :param leaky_rate: Leaky-rate
        :param spectral_radius: Spectral radius
        :param bias_scaling: Bias scaling
        :param input_scaling: Input scaling
        :param w_generator: Internal weight matrix generator
        :param win_generator: Input-reservoir weight matrix generator
        :param wbias_generator: Bias weight matrix generator
        :param nonlin_func: Non-linear function
        :param learning_algo: Learning algorithm (inv, pinv)
        :param ridge_param: Regularisation parameter
        :param with_bias: Add a bias to output ?
        :param softmax_output: Add a softmax layer at the outputs ?
        :param washout: Length of the washout period ?
        :param debug: Debug mode
        :param test_case: Test case to call for test
        :param dtype: Data type
        """
        super(LiESN, self).__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            w_generator=w_generator,
            win_generator=win_generator,
            wbias_generator=wbias_generator,
            input_scaling=input_scaling,
            nonlin_func=nonlin_func,
            learning_algo=learning_algo,
            ridge_param=ridge_param,
            softmax_output=softmax_output,
            washout=washout,
            with_bias=with_bias,
            create_rnn=False,
            create_output=True,
            debug=debug,
            test_case=test_case,
            dtype=dtype
        )

        # Generate matrices
        w, w_in, w_bias = self._generate_matrices(w_generator, win_generator, wbias_generator)

        # Recurrent layer
        self._esn_cell = LiESNCell(
            leaky_rate=leaky_rate,
            input_dim=input_dim,
            output_dim=hidden_dim,
            w=w,
            w_in=w_in,
            w_bias=w_bias,
            nonlin_func=nonlin_func,
            washout=washout,
            debug=debug,
            test_case=test_case,
            dtype=dtype
        )
    # end __init__

    # region PRIVATE

    # Create recurrent layer
    def _create_recurrent_layer(self, **kargs):
        """
        Create recurrent layer
        :param args: Recurrent layer arguments
        :return: Recurrent layer (Node)
        """
        # Recurrent layer
        return LiESNCell(**kargs)
    # end _create_recurrent_layer

    # endregion PRIVATE

# end ESNCell
