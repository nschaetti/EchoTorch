# -*- coding: utf-8 -*-
#
# File : echotorch/nn/FreeRunESN.py
# Description : Li-ESN with Feedbacks.
# Date : 28th of January, 2018
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
from .FreeRunESNCell import FreeRunESNCell
from ..Node import Node


# Li-ESN with Feedbacks
# TODO: Test
class FreeRunESN(FreeRunESNCell):
    """
    Li-ESN with Feedbacks
    """

    # Constructor
    def __init__(
            self, input_dim, hidden_dim, output_dim, leaky_rate, w_generator, win_generator, wbias_generator,
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
        super(FreeRunESN, self).__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            w_generator=w_generator,
            win_generator=win_generator,
            wbias_generator=wbias_generator,
            spectral_radius=spectral_radius,
            bias_scaling=bias_scaling,
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
            spectral_radius=spectral_radius,
            bias_scaling=bias_scaling,
            input_scaling=input_scaling,
            w=w,
            w_in=w_in,
            w_bias=w_bias,
            nonlin_func=nonlin_func,
            washout=washout,
            debug=debug,
            test_case=test_case,
            dtype=torch.float32
        )

        # Trainable elements
        self.add_trainable(self._output)
    # end __init__

    # region PROPERTIES

    # endregion PROPERTIES

    # region PUBLIC

    # endregion PUBLIC

    # region OVERRIDE

    # Finish training
    def finalize(self):
        """
        Finish training
        """
        # Train output
        self._output.finalize()

        # Set feedback matrix
        self._esn_cell.set_feedbacks(self._output.w_out)

        # In eval mode
        self.train(False)
    # end finalize

    # endregion OVERRIDE

# end FreeRunESN
