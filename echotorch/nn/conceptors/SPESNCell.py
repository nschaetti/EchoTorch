# -*- coding: utf-8 -*-
#
# File : echotorch/nn/LiESNCell.py
# Description : An Leaky-Integrated Echo State Network layer.
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
from torch.autograd import Variable
from echotorch.nn.reservoir.ESNCell import ESNCell
import matplotlib.pyplot as plt


# Self-Predicting ESN Cell
class SPESNCell(ESNCell):
    """
    Self-Predicting ESN Cell
    """

    # Loading methods
    W_LOADING = 0
    INPUTS_SIMULATION = 1
    INPUTS_RECREATION = 2

    # Constructor
    def __init__(self, w_ridge_param, w_learning_algo='inv', averaged=False, fill_left=False,
                 loading_method=W_LOADING, *args, **kwargs):
        """
        Constructor
        :param w_learning_param:
        :param w_learning_algo:
        :param averaged:
        :param fill_left:
        :param loading_method: Use W (w-loading), D (input-simulation) or R (input recreation)
        """
        # Superclass
        super(SPESNCell, self).__init__(*args, **kwargs)

        # Parameter
        self._w_ridge_param = w_ridge_param
        self._averaged = averaged
        self._w_learning_algo = w_learning_algo
        self._n_samples = 0
        self._fill_left = fill_left
        self._loading_method = loading_method

        # X covariance matrix
        self.register_buffer('xTx', Variable(torch.zeros(self._output_dim, self._output_dim, dtype=self._dtype), requires_grad=False))

        # X and Y covariance matrix
        if loading_method == SPESNCell.INPUTS_RECREATION:
            self.register_buffer('xTy', Variable(torch.zeros(self._output_dim, self._input_dim, dtype=self._dtype), requires_grad=False))
        else:
            self.register_buffer('xTy', Variable(torch.zeros(self._output_dim, self._output_dim, dtype=self._dtype), requires_grad=False))
        # end if

        # Input simulation matrix
        self.register_buffer('D', Variable(torch.zeros(self._output_dim, self._output_dim, dtype=self._dtype), requires_grad=False))

        # Input recreation matrix
        self.register_buffer('R', Variable(torch.zeros(self._input_dim, self._output_dim, dtype=self._dtype), requires_grad=False))
    # end __init__

    # region PRIVATE

    # Finalize ridge regression
    def _finalize_ridge_regression(self):
        """
        Finalize ridge regression
        """
        if self._averaged:
            # Average
            self.xTx = self.xTx / self._n_samples
            self.xTy = self.xTy / self._n_samples
        # end if

        # Debug W, Win, Wbias
        self._call_debug_point("Wstar", self.w, "SPESNCell", "_finalize_ridge_regression")
        self._call_debug_point("Win", self.w_in, "SPESNCell", "_finalize_ridge_regression")
        self._call_debug_point("Wbias", self.w_bias, "SPESNCell", "_finalize_ridge_regression")

        # Debug for xTx and xTy and ridge_param
        self._call_debug_point("xTx", self.xTx, "SPESNCell", "_finalize_ridge_regression")
        self._call_debug_point("xTy", self.xTy, "SPESNCell", "_finalize_ridge_regression")
        self._call_debug_point("w_ridge_param", self._w_ridge_param, "SPESNCell", "_finalize_ridge_regression")

        # We need to solve w = (xTx)^(-1)xTy
        # Covariance matrix xTx
        ridge_xTx = self.xTx + self._w_ridge_param * torch.eye(self.output_dim, dtype=self._dtype)

        # Debug for ridge xTx
        self._call_debug_point("ridge_xTx", ridge_xTx, "SPESNCell", "_finalize_ridge_regression")

        # Inverse / pinverse
        if self._w_learning_algo == "inv":
            inv_xTx = self._inverse("ridge_xTx", ridge_xTx, "SPESNCell", "_finalize_ridge_regression")
        elif self._w_learning_algo == "pinv":
            inv_xTx = self._pinverse("ridge_xTx", ridge_xTx, "SPESNCell", "_finalize_ridge_regression")
        else:
            raise Exception("Unknown learning method {}".format(self._learning_algo))
        # end if

        # Debug for inv_xTx
        self._call_debug_point("inv_xTx", inv_xTx, "SPESNCell", "_finalize_ridge_regression")

        return torch.mm(inv_xTx, self.xTy).t()
    # end _finalize_ridge_regression

    # Finalize W loading
    def _finalize_W_loading(self):
        """
        Finalize W loading
        """
        # Finalize ridge regression
        # w = (xTx)^(-1)xTy
        self.w = self._finalize_ridge_regression()

        # Debug for W
        self._call_debug_point("w", self.w, "SPESNCell", "_finalize_W_loading")
    # end _finalize_W_loading

    # Finalize input simulation
    def _finalize_input_simulation(self):
        """
        Finalize input simulation
        """
        # Finalize ridge regression
        # D = (xTx)^(-1)xTy
        self.D = self._finalize_ridge_regression()

        # Debug for W
        self._call_debug_point("D", self.w, "SPESNCell", "_finalize_input_simulation")
    # end _finalize_input_simulation

    # Finalize input recreation
    def _finalize_input_recreation(self):
        """
        Finalize input recreation
        """
        # Finalize ridge regression
        # R = (xTx)^(-1)xTy
        self.R = self._finalize_ridge_regression()

        # Debug for W
        self._call_debug_point("R", self.w, "SPESNCell", "_finalize_input_recreation")
    # end _finalize_input_recreation

    # Update for W loading
    def _update_W(self, states):
        """
        Update for W loading
        """
        # X (reservoir states)
        X = states[self._washout:]
        self._call_debug_point("X{}".format(self._n_samples), X, "SPESNCell", "_update_W")

        # Xold (reservoir states at t - 1))
        if self._fill_left:
            Xold = self.features(X, fill_left=states[self._washout - 1] if self._washout > 0 else None)
        else:
            Xold = self.features(X)
        # end if

        # Debug Xold
        self._call_debug_point("Xold{}".format(self._n_samples), Xold, "SPESNCell", "_update_W")

        # Y (W*x + Win*u), what we want to predict
        Y = self.targets(X)
        self._call_debug_point("Y{}".format(self._n_samples), Y, "SPESNCell", "_update_W")

        # Update covariance matrix
        self._update_covariance_matrix(Xold, Y)
    # end update_W

    # Update for input simulation
    def _update_input_simulation(self, states, inputs):
        """
        Update for input simulation
        :param states: Reservoir states
        :param inputs: Inputs
        """
        # X (reservoir states)
        X = states[self._washout:]
        self._call_debug_point("X{}".format(self._n_samples), X, "SPESNCell", "_update_input_simulation")

        # Xold (reservoir states at t - 1))
        if self._fill_left:
            Xold = self.features(X, fill_left=states[self._washout - 1] if self._washout > 0 else None)
        else:
            Xold = self.features(X)
        # end if

        # Debug Xold
        self._call_debug_point("Xold{}".format(self._n_samples), Xold, "SPESNCell", "_update_input_simulation")

        # Inputs
        U = inputs[self._washout:]

        # Y (Win*u), what we want to predict
        Y = (torch.mm(self.w_in, U.t())).t()
        self._call_debug_point("Y{}".format(self._n_samples), Y, "SPESNCell", "_update_input_simulation")

        # Update covariance matrix
        self._update_covariance_matrix(Xold, Y)
    # end update_input_simulation

    # Update for input recreation
    def _update_input_recreation(self, states, inputs):
        """
        Update for input simulation
        """
        # X (reservoir states)
        X = states[self._washout:]
        self._call_debug_point("X{}".format(self._n_samples), X, "SPESNCell", "_update_input_recreation")

        # Xold (reservoir states at t - 1))
        if self._fill_left:
            Xold = self.features(X, fill_left=states[self._washout - 1] if self._washout > 0 else None)
        else:
            Xold = self.features(X)
        # end if

        # Debug Xold
        self._call_debug_point("Xold{}".format(self._n_samples), Xold, "SPESNCell", "_update_input_recreation")

        # Inputs
        U = inputs[self._washout:]

        # Y (Win*u), what we want to predict
        Y = U
        self._call_debug_point("Y{}".format(self._n_samples), Y, "SPESNCell", "_update_input_recreation")

        # Update covariance matrix
        self._update_covariance_matrix(Xold, Y)
    # end update_input_recreation

    # Update co-variance matrix
    def _update_covariance_matrix(self, X, Y):
        """
        Update co-variance matrix
        """
        # Learn length
        learn_length = X.size(0)

        # Covariance matrices
        if self._averaged:
            self.xTx.data.add_((X.t().mm(X) / learn_length).data)
            self.xTy.data.add_((X.t().mm(Y) / learn_length).data)
        else:
            self.xTx.data.add_(X.t().mm(X).data)
            self.xTy.data.add_(X.t().mm(Y).data)
        # end if

        # Debug for xTx and xTy
        self._call_debug_point("xTx{}".format(self._n_samples), X.t().mm(X), "SPESNCell", "_update_covariance_matrix")
        self._call_debug_point("xTy{}".format(self._n_samples), X.t().mm(Y), "SPESNCell", "_update_covariance_matrix")
    # end _update_covariance_matrix

    # endregion PRIVATE

    # region OVERRIDE

    # Reset learning
    def reset(self):
        """
        Reset learning
        :return:
        """
        self.xTx.fill_(0.0)
        self.xTy.fill_(0.0)
        self.D.fill_(0.0)
        self.R.fill_(0.0)
        self._n_samples = 0

        # Training mode again
        self.train(True)
    # end reset

    # Finalize internal training
    def finalize(self):
        """
        Finalize internal training
        """
        # Loading method
        if self._loading_method == SPESNCell.W_LOADING:
            self._finalize_W_loading()
        elif self._loading_method == SPESNCell.INPUTS_SIMULATION:
            self._finalize_input_simulation()
        elif self._loading_method == SPESNCell.INPUTS_RECREATION:
            self._finalize_input_recreation()
        else:
            raise Exception("Unknown loading method {}".format(self._loading_method))
        # end if

        # Not in training mode anymore
        self.train(False)
    # end finalize

    # Compute input layer
    def _input_layer(self, ut):
        """
        Compute input layer
        :param ut: Inputs
        :return: Processed inputs
        """
        if not self.training:
            # Loading type
            if self._loading_method == SPESNCell.W_LOADING:
                return super(SPESNCell, self)._input_layer(ut)
            elif self._loading_method == SPESNCell.INPUTS_SIMULATION:
                return self.D.mv(self.hidden)
            elif self._loading_method == SPESNCell.INPUTS_RECREATION:
                return self.w_in.mv(self.R.mv(self.hidden))
            else:
                raise Exception("Unknown loading method {}".format(self._loading_method))
            # end if
        else:
            return super(SPESNCell, self)._input_layer(ut)
        # end if
    # end _input_layer

    # Hook which gets executed before the update state equation for every sample.
    def _pre_update_hook(self, inputs, forward_i, sample_i):
        """
        Hook which gets executed before the update equation for a batch
        :param inputs: Input signal.
        :param forward_i: Index of forward call
        :param sample_i: Position of the sample in the batch.
        """
        # Call debug point for inputs
        self._call_debug_point("u{}".format(self._n_samples), inputs[self._washout:], "SPESNCell", "_pre_update_hook")
        return inputs
    # end _pre_update_hook

    # Hook which gets executed before the update state equation for every timesteps.
    def _pre_step_update_hook(self, inputs, forward_i, sample_i, t):
        """
        Hook which gets executed before the update equation for every timesteps
        :param inputs: Input signal.
        :param forward_i: Index of forward call
        :param sample_i: Position of the sample in the batch.
        :param t: Timestep.
        """
        return inputs
    # end _pre_step_update_hook

    # Hook which gets executed after the update state equation for every sample.
    def _post_update_hook(self, states, inputs, forward_i, sample_i):
        """
        Hook which gets executed after the update equation for a batch
        :param states: Reservoir's states.
        :param inputs: Input signal.
        :param sample_i: Batch position.
        """
        if self.training:
            # Loading method
            if self._loading_method == SPESNCell.W_LOADING:
                self._update_W(states)
            elif self._loading_method == SPESNCell.INPUTS_SIMULATION:
                self._update_input_simulation(states, inputs)
            elif self._loading_method == SPESNCell.INPUTS_RECREATION:
                self._update_input_recreation(states, inputs)
            else:
                raise Exception("Unknown loading method {}".format(self._loading_method))
            # end if

            # Inc
            self._n_samples += 1
        # end if
        return states
    # end _post_update_hook

    # Hook which gets executed after the update state equation for every timesteps.
    def _post_step_update_hook(self, states, inputs, forward_i, sample_i, t):
        """
        Hook which gets executed after the update equation for every timesteps
        :param states: Reservoir's states.
        :param inputs: Input signal.
        :param forward_i: Index of forward call
        :param sample_i: Position of the sample in the batch
        :param t: Timestep
        """
        return states
    # end _post_step_update_hook

    # endregion OVERRIDE

    # region TARGETS

    # Features to learn from
    def features(self, X, fill_left=None):
        """
        Features
        :param X:
        :return:
        """
        # Xold (reservoir states at t - 1))
        learn_length = X.size(0)
        Xold = torch.zeros(learn_length, self.output_dim, dtype=self._dtype)
        Xold[1:, :] = X[:-1, :]
        if fill_left is not None:
            Xold[0] = fill_left
        # end if
        return Xold
    # end features

    # Targets to be learn
    def targets(self, X):
        """
        Returns targets to be learn
        :param X: Reservoir states (L, Nx)
        :return: Matrix Y to predict (L, Nx)
        """
        # Y (W*x + Win*u), what we want to predict
        learn_length = X.size(0)
        return SPESNCell.arctanh(X) - self.w_bias.repeat(learn_length, 1)
    # end targets

    # endregion TARGETS

    # region STATIC

    # Arctanh (tanh^-1)
    @staticmethod
    def arctanh(x):
        """
        Arctanh (tanh^-1)
        :param x: Tanh value to inverse
        :return: Inverse of tanh(x)
        """
        return 0.5 * torch.log((1 + x) / (1 - x))
    # end arctanh

    # endregion STATIC

# end SPESNCell
