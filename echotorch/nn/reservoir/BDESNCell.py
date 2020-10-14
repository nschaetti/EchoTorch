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
import torch.sparse
import torch
import torch.nn as nn
from echotorch.nn.reservoir.LiESNCell import LiESNCell
import numpy as np
from torch.autograd import Variable


# Bi-directional Echo State Network module
class BDESNCell(nn.Module):
    """
    Bi-directional Echo State Network module
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0,
                 w=None, w_in=None, w_bias=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None,
                 nonlin_func=torch.tanh,  leaky_rate=1.0, create_cell=True):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param hidden_dim: Hidden layer dimension
        :param spectral_radius: Reservoir's spectral radius
        :param bias_scaling: Scaling of the bias, a constant input to each neuron (default: 0, no bias)
        :param input_scaling: Scaling of the input weight matrix, default 1.
        :param w: Internal weights matrix
        :param w_in: Input-reservoir weights matrix
        :param w_bias: Bias weights matrix
        :param sparsity:
        :param input_set:
        :param w_sparsity:
        :param nonlin_func: Reservoir's activation function (tanh, sig, relu)
        """
        super(BDESNCell, self).__init__()

        # Recurrent layer
        if create_cell:
            self.esn_cell = LiESNCell(
                leaky_rate, False, input_dim, hidden_dim, spectral_radius, bias_scaling,
                input_scaling, w, w_in, w_bias, None, sparsity, input_set, w_sparsity,
                nonlin_func
            )
        # end if
    # end __init__

    # region PROPERTIES

    # Hidden weight matrix
    @property
    def w(self):
        """
        Hidden weight matrix
        :return:
        """
        return self.esn_cell.w
    # end w

    # Input matrix
    @property
    def w_in(self):
        """
        Input matrix
        :return:
        """
        return self.esn_cell.w_in
    # end w_in

    # endregion PROPERTIES

    # region PUBLIC

    # Output matrix
    def get_w_out(self):
        """
        Output matrix
        :return:
        """
        return self.output.w_out
    # end get_w_out

    # Set W
    def set_w(self, w):
        """
        Set W
        :param w:
        :return:
        """
        self.esn_cell.w = w
    # end set_w

    # endregion PUBLIC

    # region OVERRIDE

    # Reset learning
    def reset(self):
        """
        Reset learning
        :return:
        """
        # Reset output layer
        self.output.reset()

        # Training mode again
        self.train(True)
    # end reset

    # Forward
    def forward(self, u, y=None):
        """
        Forward
        :param u: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        # Forward compute hidden states
        forward_hidden_states = self.esn_cell(u)

        # Backward compute hidden states
        backward_hidden_states = self.esn_cell(Variable(torch.from_numpy(np.flip(u.data.numpy(), 1).copy())))
        backward_hidden_states = Variable(torch.from_numpy(np.flip(backward_hidden_states.data.numpy(), 1).copy()))

        return torch.cat((forward_hidden_states, backward_hidden_states), dim=2)

    # end forward

    # Finish training
    def finalize(self):
        """
        Finalize training with LU factorization
        """
        # Finalize output training
        self.output.finalize()

        # Not in training mode anymore
        self.train(False)

    # end finalize

    # Reset hidden layer
    def reset_hidden(self):
        """
        Reset hidden layer
        :return:
        """
        self.esn_cell.reset_hidden()

    # end reset_hidden

    # Get W's spectral radius
    def get_spectral_radius(self):
        """
        Get W's spectral radius
        :return: W's spectral radius
        """
        return self.esn_cell.get_spectral_raduis()
    # end spectral_radius

    # endregion OVERRIDE

# end BDESNCell
