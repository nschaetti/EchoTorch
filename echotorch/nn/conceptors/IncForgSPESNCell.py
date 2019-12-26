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
from .IncSPESNCell import IncSPESNCell
from .SPESNCell import SPESNCell
from .Conceptor import Conceptor
import matplotlib.pyplot as plt


# Self-Predicting ESN Cell with incremental-forgetting learning
class IncForgSPESNCell(IncSPESNCell):
    """
    Self-Predicting ESN Cell with incremental-forgetting learning
    """

    # Forgetting version
    FORGETTING_E = 1
    FORGETTING_A = 2

    # Constructor
    def __init__(self, lambda_param=0.0, forgetting_version=FORGETTING_A, *args, **kwargs):
        """
        Constructor
        :param lambda_param: Lambda parameters
        :param forgetting_version: Forgetting version (FORGETTING_E, FORGETTING_A)
        :param args: Arguments
        :param kwargs: Key arguments
        """
        # Superclass
        super(IncForgSPESNCell, self).__init__(
            *args,
            **kwargs
        )

        # Parameter
        self._lambda = lambda_param
        self._forgetting_version = forgetting_version
    # end __init__

    # region PRIVATE

    # Compute update matrix
    def _compute_update(self, X_old, Y):
        """
        Compute update matrix
        :param X_old: Reservoir states at time t - 1
        :param Y: Targets to predict (inputs)
        :return: Matrix update Dup
        """
        # Learn length
        learn_length = X_old.size(0)

        # We filter X- if the conceptor is not null
        if not self._conceptors.is_null():
            if self._forgetting_version == IncForgSPESNCell.FORGETTING_A:
                # The linear subspace of the reservoir state space that are not yet
                # occupied by any pattern.
                A = self._conceptors.A().C
                self._call_debug_point("A{}".format(self._n_samples), A, "IncForgSPESNCell", "_compute_update")

                # Filter old state to get only what is new
                SA_old = torch.mm(A, X_old.t()).t()
            elif self._forgetting_version == IncForgSPESNCell.FORGETTING_E:
                # The conceptor of the current pattern we want to load
                if self._w_learning_algo == "inv":
                    # Compute Conceptor for new pattern
                    C_old = Conceptor(
                        input_dim=self.output_dim,
                        aperture=self._aperture,
                        debug=self._debug,
                        dtype=self.dtype
                    )

                    # Give inputs
                    C_old(X_old)

                    # Train conceptor
                    C_old.finalize()

                    # Conflict zone
                    # Compute E = Am /\ C(m+1)
                    # The linear subspace of the reservoir state space that are
                    # occupied by pattern currently loaded and shared with
                    # the pattern we want to load.
                    E = self._conceptors.A().AND(C_old)

                    # Filter old state to get only what is in the conflict zone
                    SA_old = torch.mm(E.C, X_old.t()).t()
            else:
                raise Exception("Unknown forgetting version {}".format(self._forgetting_version))
            # end if
        else:
            # No update
            return torch.zeros(self.D.size(0), self.D.size(1), dtype=self.dtype)
        # end if

        # Debug
        self._call_debug_point("SAold{}".format(self._n_samples), SA_old, "IncForgSPESNCell", "_compute_update")

        # Targets
        if self._averaged:
            sTd = torch.mm(SA_old.t(), Y) / learn_length
        else:
            sTd = torch.mm(SA_old.t(), Y)
        # end if

        # Debug
        self._call_debug_point("sTd{}".format(self._n_samples), sTd, "IncForgSPESNCell", "_compute_update")

        # sTs
        if self._averaged:
            sTs = torch.mm(SA_old.t(), SA_old) / learn_length
        else:
            sTs = torch.mm(SA_old.t(), SA_old)
        # end if

        # Debug
        self._call_debug_point("sTs{}".format(self._n_samples), sTs, "IncForgSPESNCell", "_compute_update")

        # Ridge sTs
        ridge_sTs = sTs + math.pow(self._aperture, -2) * torch.eye(self._output_dim)

        # Debug
        self._call_debug_point("ridge_sTs{}".format(self._n_samples), ridge_sTs, "IncForgSPESNCell", "_compute_update")

        # Inverse / pinverse
        if self._w_learning_algo == "inv":
            inv_sTs = self._inverse("ridge_sTs", ridge_sTs, "IncForgSPESNCell", "_compute_update")
        elif self._w_learning_algo == "pinv":
            inv_sTs = self._pinverse("ridge_sTs", ridge_sTs, "IncForgSPESNCell", "_compute_update")
        else:
            raise Exception("Unknown learning method {}".format(self._learning_algo))
        # end if

        # Debug
        self._call_debug_point("inv_sTs{}".format(self._n_samples), inv_sTs, "IncForgSPESNCell", "_compute_update")

        # Compute the increment for matrix D
        return torch.mm(inv_sTs, sTd).t()
    # end _compute_update

    # Compute conflict zone
    def _compute_conflict_zone(self, X):
        # Compute Conceptor for new pattern
        C = Conceptor(
            input_dim=self.output_dim,
            aperture=self._aperture,
            debug=self._debug,
            dtype=self.dtype
        )

        # Give inputs
        C(X)

        # Train conceptor
        C.finalize()

        # Conflict zone
        # Compute E = Am /\ C(m+1)
        # The linear subspace of the reservoir state space that are
        # occupied by pattern currently loaded and shared with
        # the pattern we want to load.
        E = self._conceptors.A().AND(C)

        return C, E
    # end if

    # Update input simulation matrix D
    def _update_D_loading(self, states, inputs):
        """
        Update input simulation matrix D
        """
        # Get X and U
        X, X_old, U = self._compute_XU(states, inputs)

        # Compute conflict zone
        if self._forgetting_version == IncForgSPESNCell.FORGETTING_E:
            Cm1, E = self._compute_conflict_zone(X)
        # end if

        # Targets : what cannot be predicted by the
        # current matrix D.
        Y = (torch.mm(self.w_in, U.t()) - torch.mm(self.D, X_old.t())).t()

        # Targets : what cannot be predicted by the
        # current matrix D/R.
        self._call_debug_point("Td{}".format(self._n_samples), Y, "IncSPESNCell", "_update_D_loading")

        # Compute the increment for matrix D
        Dinc = self._compute_increment(X_old, Y)

        # Compute the update for matrix D
        Dup = self._compute_update(X_old, Y)

        # Debug
        self._call_debug_point("Dinc{}".format(self._n_samples), Dinc, "IncSPESNCell", "_update_D_loading")

        # Increment D
        if self._forgetting_version == IncForgSPESNCell.FORGETTING_A:
            self.D = (1.0 - self._lambda) * self.D + self._lambda * Dup + Dinc
        else:
            self.D = torch.mm(Cm1.NOT().C, self.D) + \
                     (1.0 - self._lambda) * torch.mm(E, self.D) + \
                     self._lambda * Dup + Dinc
        # end if

        # Debug
        self._call_debug_point("D{}".format(self._n_samples), self.D, "IncSPESNCell", "_update_D_loading")
    # end _update_D_loading

    # Update input recreation matrix R
    def _update_R_loading(self, states, inputs):
        """
        Update input recreation matrix R
        """
        # Get X and U
        X, X_old, U = self._compute_XU(states, inputs)

        # Targets : what cannot be predicted by the
        # current matrix D.
        Y = (U.t() - torch.mm(self.R, X_old.t())).t()
        self._call_debug_point("Td{}".format(self._n_samples), Y, "IncSPESNCell", "_update_R_loading")

        # Compute the increment for matrix D
        Rinc = self._compute_increment(X_old, Y)

        # Compute the update for matrix R
        Rup = self._compute_update(X_old, Y)

        # Debug
        self._call_debug_point("Rinc{}".format(self._n_samples), Rinc, "IncSPESNCell", "_update_R_loading")

        # Increment R
        self.R = (1.0 - self._lambda) * self.R + self._lambda * Rup + Rinc

        # Debug
        self._call_debug_point("R{}".format(self._n_samples), self.R, "IncSPESNCell", "_update_R_loading")
    # end _update_R_loading

    # endregion PRIVATE

    # region OVERRIDE

    # endregion OVERRIDE

# end IncSPESNCell
