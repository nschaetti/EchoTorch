# -*- coding: utf-8 -*-
#
# File : echotorch/nn/IncSPESNCell.py
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
# Copyright Nils Schaetti <nils.schaetti@unine.ch>

"""
Created on 5th November 2019
@author: Nils Schaetti
"""

# Imports
import math
import torch
from torch.autograd import Variable
from echotorch.nn.reservoir.ESNCell import ESNCell
from .SPESNCell import SPESNCell
import matplotlib.pyplot as plt


# Self-Predicting ESN Cell with incremental learning
class IncSPESNCell(SPESNCell):
    """
    Self-Predicting ESN Cell with incremental learning
    """

    # Constructor
    def __init__(self, conceptors, aperture, *args, **kwargs):
        """
        Constructor
        :param conceptors: List of conceptors as ConceptorSet
        :param aperture:
        :param args: Arguments
        :param kwargs: Key arguments
        """
        # Superclass
        super(IncSPESNCell, self).__init__(*args, **kwargs)

        # Parameter
        self._conceptors = conceptors
        self._aperture = aperture

        # Set it as buffer
        self.register_buffer(
            'D',
            Variable(torch.zeros(self._output_dim, self._output_dim, dtype=self._dtype), requires_grad=False)
        )
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
        self.D.data.fill_(0.0)

        # Training mode again
        self.train(True)
    # end reset

    ##################
    # OVERLOAD
    ##################

    # Compute input layer
    def _input_layer(self, ut):
        """
        Compute input layer
        :param ut: Inputs
        :return: Processed inputs
        """
        if not self.training:
            return self.D.mv(self.hidden)
        else:
            return self.win.mv(ut)
        # end if
    # end _input_layer

    # Hook which gets executed after the update state equation for every sample.
    def _post_update_hook(self, states, inputs, forward_i, sample_i):
        """
        Hook which gets executed after the update equation for a batch
        :param states: Reservoir's states.
        :param inputs: Input signal.
        :param sample_i: Batch position.
        """
        if self.training:
            # Get old states as features
            X_old = self.features(states)

            # Learn length
            learn_length = X_old.size(0)

            # Targets : what cannot be predicted by the
            # current matrix D.
            Td = torch.mm(self.w_in, inputs.reshape(1, -1)) - torch.mm(self.D, X_old)

            # The linear subspace of the reservoir state space that are not yet
            # occupied by any pattern.
            F = self._conceptors.F()

            # Filter old state to get only what is new
            S_old = torch.mm(F, X_old)

            # Targets
            if self._averaged:
                sTd = torch.mm(S_old.t(), Td) / learn_length
            else:
                sTd = torch.mm(S_old.t(), Td)
            # end if

            # sTs
            if self._averaged:
                sTs = torch.mm(S_old.t(), S_old) / learn_length
            else:
                sTs = torch.mm(S_old.t(), S_old)
            # end if

            # Ridge sTs
            ridge_sTs = sTs + math.pow(self._aperture, -2) * torch.eye(self._output_dim)

            # Inverse / pinverse
            if self._w_learning_algo == "inv":
                inv_sTs = self._inverse("ridge_xTx", ridge_sTs)
            elif self._w_learning_algo == "pinv":
                inv_sTs = self._pinverse("ridge_xTx", ridge_sTs)
            else:
                raise Exception("Unknown learning method {}".format(self._learning_algo))
            # end if

            # Compute the increment for matrix D
            Dinc = torch.mm(inv_sTs, sTd).t()

            # Increment D
            self.D += Dinc
        # end if

        return states
    # end _post_update_hook

# end IncSPESNCell
