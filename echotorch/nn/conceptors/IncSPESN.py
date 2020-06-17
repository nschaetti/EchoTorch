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
from ..linear import IncRRCell, IncForgRRCell
from .IncSPESNCell import IncSPESNCell
from .IncForgSPESNCell import IncForgSPESNCell
from .SPESNCell import SPESNCell
from ..reservoir import ESN
from ..Node import Node


# Self-Predicting Echo State Network module with incremental learning
class IncSPESN(ESN):
    """
    Self-Predicting Echo State Network module with incremental learning
    """
    # region BODY

    # Constructor
    def __init__(self, input_dim, hidden_dim, output_dim, conceptors, w_generator, win_generator, wbias_generator,
                 input_scaling=1.0, nonlin_func=torch.tanh, learning_algo_wout='pinv', learning_algo_w='inv',
                 ridge_param_inc=0.01, ridge_param_up=0.01, ridge_param_wout=0.01, ridge_param_wout_inc=0.01,
                 ridge_param_wout_up=0.01, aperture=1, with_bias=False, softmax_output=False, washout=0,
                 cell_averaged=True, output_averaged=True, fill_left=False, loading_method=SPESNCell.INPUTS_SIMULATION,
                 incremental_forgetting=False, forgetting_lambda=0.0,
                 forgetting_version=IncForgSPESNCell.FORGETTING_VERSION1, forgetting_threshold=0.95,
                 debug=Node.NO_DEBUG, test_case=None, dtype=torch.float32):
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
        :param learning_algo_wout: Output learning method (inv, pinv)
        :param learning_algo_w: Loading method (inv, pinv)
        :param ridge_param: Ridge parameter
        :param with_bias: Add a bias to the output layer ?
        :param softmax_output: Add a softmax output layer
        :param washout: Washout period (ignore timesteps at the beginning of each sample)
        :param loading_method:
        :param incremental_forgetting: Active incremental forgetting ?
        :param forgetting_threshold: Forgetting threshold
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
            create_rnn=False,
            create_output=False,
            debug=debug,
            washout=washout,
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
        self._incremental_forgetting = incremental_forgetting
        self._dtype = dtype

        # Generate matrices
        w, w_in, w_bias = self._generate_matrices(w_generator, win_generator, wbias_generator)

        # Recurrent layer
        if incremental_forgetting:
            self._esn_cell = IncForgSPESNCell(
                input_dim=input_dim,
                output_dim=hidden_dim,
                conceptors=conceptors,
                w=w,
                w_in=w_in,
                w_bias=w_bias,
                input_scaling=input_scaling,
                nonlin_func=nonlin_func,
                w_learning_algo=learning_algo_w,
                aperture=aperture,
                ridge_param_inc=ridge_param_inc,
                ridge_param_up=ridge_param_up,
                washout=washout,
                fill_left=fill_left,
                averaged=cell_averaged,
                loading_method=loading_method,
                lambda_param=forgetting_lambda,
                forgetting_version=forgetting_version,
                forgetting_threshold=forgetting_threshold,
                debug=debug,
                test_case=test_case,
                dtype=dtype
            )

            # Output layer
            self._output = IncForgRRCell(
                input_dim=hidden_dim,
                output_dim=output_dim,
                ridge_param_inc=ridge_param_wout_inc,
                ridge_param_up=ridge_param_wout_up,
                aperture=aperture,
                with_bias=with_bias,
                learning_algo=learning_algo_wout,
                softmax_output=softmax_output,
                conceptors=conceptors,
                averaged=output_averaged,
                lambda_param=forgetting_lambda,
                forgetting_version=forgetting_version,
                forgetting_threshold=forgetting_threshold,
                debug=debug,
                test_case=test_case,
                dtype=dtype
            )
        else:
            self._esn_cell = IncSPESNCell(
                input_dim=input_dim,
                output_dim=hidden_dim,
                conceptors=conceptors,
                w=w,
                w_in=w_in,
                w_bias=w_bias,
                input_scaling=input_scaling,
                nonlin_func=nonlin_func,
                w_learning_algo=learning_algo_w,
                aperture=aperture,
                washout=washout,
                fill_left=fill_left,
                averaged=cell_averaged,
                loading_method=loading_method,
                debug=debug,
                test_case=test_case,
                dtype=dtype
            )

            # Output layer
            self._output = IncRRCell(
                input_dim=hidden_dim,
                output_dim=output_dim,
                ridge_param=ridge_param_wout,
                with_bias=with_bias,
                learning_algo=learning_algo_wout,
                softmax_output=softmax_output,
                conceptors=conceptors,
                averaged=output_averaged,
                debug=debug,
                test_case=test_case,
                dtype=dtype
            )
        # end if
    # end __init__

    # endregion BODY
# end IncSPESN
