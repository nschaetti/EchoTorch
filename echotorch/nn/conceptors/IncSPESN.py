# -*- coding: utf-8 -*-
#
# File : echotorch/nn/conceptors/IncSPESN.py
# Description : Self-Predicting ESN with incremental learning.
# Date : 5th of November, 2019
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
Created on 5th November 2019
@author: Nils Schaetti
"""

# Imports
import torch
from echotorch.nn.linear.RRCell import RRCell
from .SPESN import SPESN
from .IncSPESNCell import IncSPESNCell
from ..reservoir import ESN
from ..Node import Node


# Self-Predicting Echo State Network module with incremental learning
class IncSPESN(SPESN):
    """
    Self-Predicting Echo State Network module with incremental learning
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, output_dim, conceptors, w_generator, win_generator, wbias_generator,
                 spectral_radius=0.9, bias_scaling=1.0, input_scaling=1.0, nonlin_func=torch.tanh, learning_algo='inv',
                 w_learning_algo='inv', ridge_param=0.000001, aperture=0.0001, with_bias=True,
                 softmax_output=False, washout=0, debug=Node.NO_DEBUG, test_case=None, dtype=torch.float32):
        """
        Constructor
        :param input_dim: Input feature space dimension
        :param hidden_dim: Hidden space dimension
        :param output_dim: Output space dimension
        :param conceptors: Conceptors as a ConceptorSet object
        :param w_generator: Internal weight matrix generator
        :param win_generator: Input-output weight matrix generator
        :param wbias_generator: Bias matrix generator
        :param spectral_radius: Spectral radius
        :param bias_scaling: Bias scaling
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
        super(IncSPESN, self).__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            w_generator=w_generator,
            win_generator=win_generator,
            wbias_generator=wbias_generator,
            ridge_param=ridge_param,
            learning_algo=learning_algo,
            softmax_output=softmax_output,
            debug=debug,
            test_case=test_case
        )

        # Generate matrices
        w, w_in, w_bias = self._generate_matrices(w_generator, win_generator, wbias_generator)

        # Recurrent layer
        self._esn_cell = IncSPESNCell(
            input_dim=input_dim,
            output_dim=hidden_dim,
            conceptors=conceptors,
            w=w,
            w_in=w_in,
            w_bias=w_bias,
            spectral_radius=spectral_radius,
            bias_scaling=bias_scaling,
            input_scaling=input_scaling,
            nonlin_func=nonlin_func,
            w_learning_algo=w_learning_algo,
            aperture=aperture,
            washout=washout,
            debug=debug,
            test_case=test_case,
            dtype=dtype
        )
    # end __init__

# end IncSPESN
