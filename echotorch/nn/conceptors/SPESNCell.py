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

    # Constructor
    def __init__(self, w_ridge_param, w_learning_algo='inv', averaged=False, *args, **kwargs):
        """
        Constructor
        """
        # Superclass
        super(SPESNCell, self).__init__(*args, **kwargs)

        # Parameter
        self._w_ridge_param = w_ridge_param
        self._averaged = averaged
        self._w_learning_algo = w_learning_algo
        self._n_samples = 0

        # Set it as buffer
        self.register_buffer('xTx', Variable(torch.zeros(self._output_dim, self._output_dim, dtype=self._dtype), requires_grad=False))
        self.register_buffer('xTy', Variable(torch.zeros(self._output_dim, self._output_dim, dtype=self._dtype), requires_grad=False))
    # end __init__

    ##################
    # PUBLIC
    ##################

    # Reset learning
    def reset(self):
        """
        Reset learning
        :return:
        """
        self.xTx.data.fill_(0.0)
        self.xTy.data.fill_(0.0)
        self._n_samples = 0

        # Training mode again
        self.train(True)
    # end reset

    # Finalize internal training
    def finalize(self):
        """
        Finalize internal training
        """
        if self._averaged:
            # Average
            self.xTx = self.xTx / self._n_samples
            self.xTy = self.xTy / self._n_samples
        # end if

        # Debug W, Win, Wbias
        self._call_debug_point("Wstar", self.w, "SPESNCell", "finalize")
        self._call_debug_point("Win", self.w_in, "SPESNCell", "finalize")
        self._call_debug_point("Wbias", self.w_bias, "SPESNCell", "finalize")

        # Debug for xTx and xTy and ridge_param
        self._call_debug_point("xTx", self.xTx, "SPESNCell", "finalize")
        self._call_debug_point("xTy", self.xTy, "SPESNCell", "finalize")
        self._call_debug_point("w_ridge_param", self._w_ridge_param, "SPESNCell", "finalize")

        # We need to solve w = (xTx)^(-1)xTy
        # Covariance matrix xTx
        ridge_xTx = self.xTx + self._w_ridge_param * torch.eye(self.output_dim, dtype=self._dtype)

        # Debug for ridge xTx
        self._call_debug_point("ridge_xTx", ridge_xTx, "SPESNCell", "finalize")

        # Inverse / pinverse
        if self._w_learning_algo == "inv":
            inv_xTx = self._inverse("ridge_xTx", ridge_xTx)
        elif self._w_learning_algo == "pinv":
            inv_xTx = self._pinverse("ridge_xTx", ridge_xTx)
        else:
            raise Exception("Unknown learning method {}".format(self._learning_algo))
        # end if

        # Debug for inv_xTx
        self._call_debug_point("inv_xTx", inv_xTx, "SPESNCell", "finalize")

        # w = (xTx)^(-1)xTy
        self.w.data = torch.mm(inv_xTx, self.xTy).t().data

        # Debug for W
        self._call_debug_point("w", self.w, "SPESNCell", "finalize")

        # Not in training mode anymore
        self.train(False)
    # end finalize

    # region OVERRIDE

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
            # X (reservoir states)
            X = states[self._washout:]
            self._call_debug_point("X{}".format(self._n_samples), X, "SPESNCell", "_post_update_hook")

            # Learn length
            learn_length = X.size(0)

            # Xold (reservoir states at t - 1))
            Xold = self.features(X, fill_left=states[self._washout-1] if self._washout > 0 else None)
            self._call_debug_point("Xold{}".format(self._n_samples), Xold, "SPESNCell", "_post_update_hook")

            # Y (W*x + Win*u), what we want to predict
            Y = self.targets(X)
            self._call_debug_point("Y{}".format(self._n_samples), Y, "SPESNCell", "_post_update_hook")

            # Covariance matrices
            if self._averaged:
                self.xTx.data.add_((Xold.t().mm(Xold) / learn_length).data)
                self.xTy.data.add_((Xold.t().mm(Y) / learn_length).data)
            else:
                self.xTx.data.add_(Xold.t().mm(Xold).data)
                self.xTy.data.add_(Xold.t().mm(Y).data)
            # end if

            # Debug for xTx and xTy
            self._call_debug_point("xTx{}".format(self._n_samples), Xold.t().mm(Xold), "SPESNCell", "_post_update_hook")
            self._call_debug_point("xTy{}".format(self._n_samples), Xold.t().mm(Y), "SPESNCell", "_post_update_hook")

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
