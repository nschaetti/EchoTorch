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
# Copyright Nils Schaetti <nils.schaetti@unine.ch>

"""
Created on 26 January 2018
@author: Nils Schaetti
"""

import torch
from .LiESNCell import LiESNCell
from echotorch.nn.reservoir.ESN import ESN


# Leaky-Integrated Echo State Network module
class LiESN(ESN):
    """
    Leaky-Integrated Echo State Network module
    """

    # Constructor
    def __init__(self, leaky_rate, input_dim, hidden_dim, output_dim, w_generator, win_generator, wbias_generator,
                 spectral_radius=0.9, bias_scaling=1.0, input_scaling=1.0, nonlin_func=torch.tanh, learning_algo='inv',
                 ridge_param=0.0, with_bias=True, softmax_output=False, washout=0, dtype=torch.float32):
        """
        Constructor
        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        :param spectral_radius:
        :param bias_scaling:
        :param input_scaling:
        :param w:
        :param w_in:
        :param w_bias:
        :param sparsity:
        :param input_set:
        :param w_sparsity:
        :param nonlin_func:
        :param learning_algo:
        :param ridge_param:
        :param leaky_rate:
        :param train_leaky_rate:
        :param feedbacks:
        """
        super(LiESN, self).__init__(
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
            dtype=dtype
        )

        # Generate matrices
        w, w_in, w_bias = self._generate_matrices(w_generator, win_generator, wbias_generator)

        # Recurrent layer
        self.esn_cell = LiESNCell(
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
            dtype=torch.float32
        )
    # end __init__

    ###############################################
    # PROPERTIES
    ###############################################

    ###############################################
    # PUBLIC
    ###############################################

    ###############################################
    # PRIVATE
    ###############################################

# end ESNCell
