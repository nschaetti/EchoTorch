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

        # We need to solve w = (xTx)^(-1)xTy
        # Covariance matrix xTx
        ridge_xTx = self.xTx + self._w_ridge_param * torch.eye(self._input_dim, dtype=self._dtype)

        # Inverse / pinverse
        if self._w_learning_algo == "inv":
            inv_xTx = ridge_xTx.inverse()
        elif self._w_learning_algo == "pinv":
            inv_xTx = ridge_xTx.pinverse()
        else:
            raise Exception("Unknown learning method {}".format(self._learning_algo))
        # end if

        # w = (xTx)^(-1)xTy
        self.w.data = torch.mm(inv_xTx, self.xTy).data

        # Not in training mode anymore
        self.train(False)
    # end finalize

    ##################
    # OVERLOAD
    ##################

    # Hook which gets executed before the update state equation for every sample.
    def _pre_update_hook(self, inputs, sample_i):
        """
        Hook which gets executed before the update equation for a batch
        :param inputs: Input signal.
        :param sample_i: Batch position.
        """
        pass
        return inputs
    # end _pre_update_hook

    # Hook which gets executed before the update state equation for every timesteps.
    def _pre_step_update_hook(self, inputs, t):
        """
        Hook which gets executed before the update equation for every timesteps
        :param inputs: Input signal.
        :param t: Timestep.
        """
        return inputs
    # end _pre_step_update_hook

    # Hook which gets executed after the update state equation for every sample.
    def _post_update_hook(self, states, inputs, sample_i):
        """
        Hook which gets executed after the update equation for a batch
        :param states: Reservoir's states.
        :param inputs: Input signal.
        :param sample_i: Batch position.
        """
        if self.training:
            # X (reservoir states)
            X = states[self._washout:]

            # Learn length
            learn_length = X.size(0)

            # Xold (reservoir states at t - 1))
            Xold = torch.zeros(learn_length, self.output_dim, dtype=self._dtype)
            Xold[1:, :] = states[self._washout:-1]

            # Y (W*x + Win*u), what we want to predict
            Y = SPESNCell.arctanh(X) - self.w_bias.repeat(learn_length, 1)

            # Covariance matrices
            if self._averaged:
                self.xTx.data.add_((Xold.t().mm(Xold) / learn_length).data)
                self.xTy.data.add_((Xold.t().mm(Y) / learn_length).data)
            else:
                self.xTx.data.add_(Xold.t().mm(Xold).data)
                self.xTy.data.add_(Xold.t().mm(Y).data)
            # end if

            # Inc
            self._n_samples += 1
        # end if
        return states
    # end _post_update_hook

    # Hook which gets executed after the update state equation for every timesteps.
    def _post_step_update_hook(self, states, inputs, t):
        """
        Hook which gets executed after the update equation for every timesteps
        :param states: Reservoir's states.
        :param inputs: Input signal.
        :param t: Timestep.
        """
        return states
    # end _post_step_update_hook

    ##################
    # STATIC
    ##################

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

# end SPESNCell
