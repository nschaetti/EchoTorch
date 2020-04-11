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
    def __init__(self, conceptors, aperture, averaged=True, *args, **kwargs):
        """
        Constructor
        :param conceptors: List of conceptors as ConceptorSet
        :param aperture:
        :param args: Arguments
        :param kwargs: Key arguments
        """
        # Superclass
        super(IncSPESNCell, self).__init__(
            w_ridge_param=0,
            averaged=averaged,
            *args,
            **kwargs
        )

        # Parameter
        self._conceptors = conceptors
        self._aperture = aperture

        # Input simulation matrix increment
        self.register_buffer(
            'Dinc',
            Variable(torch.zeros(self._output_dim, self._output_dim, dtype=self._dtype), requires_grad=False)
        )

        # Input recreation matrix increment
        self.register_buffer(
            'Rinc',
            Variable(torch.zeros(self._input_dim, self._output_dim, dtype=self._dtype), requires_grad=False)
        )
    # end __init__

    # region PRIVATE

    # Compute increment matrix
    def _compute_increment(self, X, Y, ridge_param, F=None):
        """
        Compute increment matrix
        """
        # Learn length
        learn_length = X.size(0)

        # We filter X- if the conceptor is not null
        if not self._conceptors.is_null():
            # The linear subspace of the reservoir state space that are not yet
            # occupied by any pattern.
            if F is None:
                F = self._conceptors.F()
            # end if
            self._call_debug_point("F{}".format(self._n_samples), F, "IncSPESNCell", "_compute_increment")

            # Filter old state to get only what is new
            S_old = torch.mm(F, X.t()).t()
        else:
            # No filter
            S_old = X
        # end if

        # Debug
        self._call_debug_point("Sold{}".format(self._n_samples), S_old, "IncSPESNCell", "_compute_increment")

        # Targets
        if self._averaged:
            sTd = torch.mm(S_old.t(), Y) / learn_length
        else:
            sTd = torch.mm(S_old.t(), Y)
        # end if

        # Debug
        self._call_debug_point("sTd{}".format(self._n_samples), sTd, "IncSPESNCell", "_compute_increment")

        # sTs
        if self._averaged:
            sTs = torch.mm(S_old.t(), S_old) / learn_length
        else:
            sTs = torch.mm(S_old.t(), S_old)
        # end if

        # Debug
        self._call_debug_point("sTs{}".format(self._n_samples), sTs, "IncSPESNCell", "_compute_increment")

        # Ridge sTs
        ridge_sTs = sTs + ridge_param * torch.eye(self._output_dim)

        # Debug
        self._call_debug_point("ridge_sTs{}".format(self._n_samples), ridge_sTs, "IncSPESNCell", "_compute_increment")

        # Inverse / pinverse
        if self._w_learning_algo == "inv":
            inv_sTs = self._inverse("ridge_sTs", ridge_sTs, "IncSPESNCell", "_compute_increment")
        elif self._w_learning_algo == "pinv":
            inv_sTs = self._pinverse("ridge_sTs", ridge_sTs, "IncSPESNCell", "_compute_increment")
        else:
            raise Exception("Unknown learning method {}".format(self._learning_algo))
        # end if

        # Debug
        self._call_debug_point("inv_sTs{}".format(self._n_samples), inv_sTs, "IncSPESNCell", "_compute_increment")

        # Compute the increment for matrix D
        return torch.mm(inv_sTs, sTd).t()
    # end _compute_increment

    # Update input simulation matrix D
    def _update_D_loading(self, states, inputs):
        """
        Update input simulation matrix D
        """
        # Get X and U
        _, X_old, U = self._compute_XU(states, inputs)

        # Targets : what cannot be predicted by the
        # current matrix D.
        Y = (torch.mm(self.w_in, U.t()) - torch.mm(self.D, X_old.t())).t()

        # Targets : what cannot be predicted by the
        # current matrix D/R.
        self._call_debug_point("Td{}".format(self._n_samples), Y, "IncSPESNCell", "_update_D_loading")

        # Compute the increment for matrix D
        self.Dinc = self._compute_increment(X_old, Y, math.pow(self._aperture, -2))

        # Debug
        self._call_debug_point("Dinc{}".format(self._n_samples), self.Dinc, "IncSPESNCell", "_update_D_loading")

        # Increment D
        self.D += self.Dinc

        # Debug
        self._call_debug_point("D{}".format(self._n_samples), self.D, "IncSPESNCell", "_update_D_loading")
    # end _update_D_loading

    # Update input recreation matrix R
    def _update_R_loading(self, states, inputs):
        """
        Update input recreation matrix R
        """
        # Get X and U
        _, X_old, U = self._compute_XU(states, inputs)

        # Targets : what cannot be predicted by the
        # current matrix D.
        Y = (U.t() - torch.mm(self.R, X_old.t())).t()
        self._call_debug_point("Td{}".format(self._n_samples), Y, "IncSPESNCell", "_update_R_loading")

        # Compute the increment for matrix D
        self.Rinc = self._compute_increment(X_old, Y, math.pow(self._aperture, -2))

        # Debug
        self._call_debug_point("Rinc{}".format(self._n_samples), self.Rinc, "IncSPESNCell", "_update_R_loading")

        # Increment R
        self.R += self.Rinc

        # Debug
        self._call_debug_point("R{}".format(self._n_samples), self.R, "IncSPESNCell", "_update_R_loading")
    # end _update_R_loading

    # Compute X and U
    def _compute_XU(self, states, inputs):
        """
        Compute X and U
        :param states: Reservoir states
        :param inputs: Reservoir inputs
        """
        # X (reservoir states)
        X = states[self._washout:]
        self._call_debug_point("X{}".format(self._n_samples), X, "IncSPESNCell", "_compute_XU")

        # Get old states as features
        if self._fill_left:
            X_old = self.features(X, fill_left=states[self._washout - 1] if self._washout > 0 else None)
        else:
            X_old = self.features(X)
        # end if

        # Debug Xold
        self._call_debug_point("Xold{}".format(self._n_samples), X_old, "IncSPESNCell", "_compute_XU")

        # Inputs
        U = inputs[self._washout:]

        return X, X_old, U
    # end _compute_XU

    # endregion PRIVATE

    # region OVERRIDE

    # Finalize training
    def finalize(self):
        """
        Finalize training
        """
        # Debug W, Win, Wbias
        self._call_debug_point("Wstar", self.w, "IncSPESNCell", "finalize")
        self._call_debug_point("Win", self.w_in, "IncSPESNCell", "finalize")
        self._call_debug_point("Wbias", self.w_bias, "IncSPESNCell", "finalize")

        # Not in training mode anymore
        self.train(False)
    # end reset

    # Compute input layer
    def _input_layer(self, ut):
        """
        Compute input layer
        :param ut: Inputs
        :return: Processed inputs
        """
        if not self.training:
            # Loading type
            if self._loading_method == SPESNCell.INPUTS_SIMULATION:
                return self.D.mv(self.hidden)
            elif self._loading_method == SPESNCell.INPUTS_RECREATION:
                return self.w_in.mv(self.R.mv(self.hidden))
            else:
                raise Exception("Unknown loading method {} for incremental loading".format(self._loading_method))
            # end if
        else:
            return super(SPESNCell, self)._input_layer(ut)
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
        # Update D if in training mode
        if self.training:
            # Loading method
            if self._loading_method == SPESNCell.INPUTS_SIMULATION:
                self._update_D_loading(states, inputs)
            elif self._loading_method == SPESNCell.INPUTS_RECREATION:
                self._update_R_loading(states, inputs)
            else:
                raise Exception("Unknown loading method {} for incremental loading".format(self._loading_method))
            # end if

            # Inc
            self._n_samples += 1
        # end if

        return states
    # end _post_update_hook

    # endregion OVERRIDE

# end IncSPESNCell
