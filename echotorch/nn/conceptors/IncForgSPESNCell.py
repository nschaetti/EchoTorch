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
    FORGETTING_VERSION1 = 1
    FORGETTING_VERSION2 = 2
    FORGETTING_VERSION3 = 3
    FORGETTING_VERSION4 = 4
    FORGETTING_VERSION5 = 5

    # Constructor
    def __init__(self, lambda_param=0.0, forgetting_threshold=0.95, forgetting_version=FORGETTING_VERSION1,
                 *args, **kwargs):
        """
        Constructor
        :param lambda_param: Lambda parameters
        :param forgetting_threshold: Forgetting threshold
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
        self._forgetting_threshold = forgetting_threshold

        # Input simulation matrix update
        self.register_buffer(
            'Dup',
            Variable(torch.zeros(self._output_dim, self._output_dim, dtype=self._dtype), requires_grad=False)
        )

        # Input simulation matrix update
        self.register_buffer(
            'Dnew',
            Variable(torch.zeros(self._output_dim, self._output_dim, dtype=self._dtype), requires_grad=False)
        )

        # Input recreation matrix update
        self.register_buffer(
            'Rup',
            Variable(torch.zeros(self._input_dim, self._output_dim, dtype=self._dtype), requires_grad=False)
        )

        # Input recreation matrix update
        self.register_buffer(
            'Rnew',
            Variable(torch.zeros(self._input_dim, self._output_dim, dtype=self._dtype), requires_grad=False)
        )
    # end __init__

    # region PRIVATE

    # Compute update matrix
    def _compute_update(self, X_old, Y, E, C):
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
            # The linear subspace of the reservoir state space
            # occupied by loaded patterns.
            A = self._conceptors.A().C
            self._call_debug_point("A{}".format(self._n_samples), A, "IncForgSPESNCell", "_compute_update")

            # For each version
            if self._forgetting_version == IncForgSPESNCell.FORGETTING_VERSION1:
                # Filter old state to get only what is new in that space.
                # A * X(t-1)
                S_old = torch.mm(A, X_old.t()).t()
            elif self._forgetting_version == IncForgSPESNCell.FORGETTING_VERSION2:
                # The linear subspace of conflict between loaded patterns and the new one.
                # E * X(t-1)
                S_old = torch.mm(E.C, X_old.t()).t()
            elif self._forgetting_version == IncForgSPESNCell.FORGETTING_VERSION3:
                # The linear subspace of conflict between loaded patterns and the new one.
                # C * A * X(t-1)
                S_old = torch.mm(torch.mm(C.C, A), X_old.t()).t()
            elif self._forgetting_version == IncForgSPESNCell.FORGETTING_VERSION4:
                # Use the whole space as features
                # X(t-1)
                S_old = X_old
            elif self._forgetting_version == IncForgSPESNCell.FORGETTING_VERSION5:
                # Use the space occupied by the other patterns
                # A * X(t-1)
                S_old = torch.mm(A, X_old.t()).t()
            # end if
        else:
            # No update
            if self._loading_method == IncSPESNCell.INPUTS_SIMULATION:
                return torch.zeros(self.D.size(0), self.D.size(1), dtype=self.dtype)
            elif self._loading_method == IncSPESNCell.INPUTS_RECREATION:
                return torch.zeros(self.R.size(0), self.R.size(1), dtype=self.dtype)
            else:
                raise Exception("Unknown loading method {}".format(self._loading_method))
            # end if
        # end if

        # Debug
        self._call_debug_point("SAold{}".format(self._n_samples), S_old, "IncForgSPESNCell", "_compute_update")

        # Targets
        if self._averaged:
            sTd = torch.mm(S_old.t(), Y) / learn_length
        else:
            sTd = torch.mm(S_old.t(), Y)
        # end if

        # Debug
        self._call_debug_point("sTd{}".format(self._n_samples), sTd, "IncForgSPESNCell", "_compute_update")

        # sTs
        if self._averaged:
            sTs = torch.mm(S_old.t(), S_old) / learn_length
        else:
            sTs = torch.mm(S_old.t(), S_old)
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

    # Compute conceptor of the current pattern
    def _compute_conceptor(self, X):
        """
        Compute
        """
        # Compute Conceptor for new pattern
        C = Conceptor(
            input_dim=self.output_dim,
            aperture=self._aperture,
            debug=self._debug,
            dtype=self.dtype
        )

        # Give inputs to be learned
        C(X)

        # Train conceptor
        C.finalize()

        # NOT C
        NOT_C = Conceptor.operator_NOT(C)

        # Space occupied by currently loaded
        # patterns.
        A = self._conceptors.A()

        # Conflict zone
        E = Conceptor.operator_AND(A, C)

        # Demilitarized zone
        M = Conceptor.operator_AND(A, NOT_C)

        return C, NOT_C, E, M
    # end if

    # Update input simulation matrix D
    def _update_D_loading(self, states, inputs):
        """
        Update input simulation matrix D
        """
        # Get X and U
        X, X_old, U = self._compute_XU(states, inputs)

        # Compute conflict zone
        C, NC, E, M = self._compute_conceptor(X)

        # Targets for the increment matrix Dinc
        Yinc = (torch.mm(self.w_in, U.t()) - torch.mm(self.D, X_old.t())).t()

        # Compute the increment and update for matrix D
        self.Dinc = self._compute_increment(X_old, Yinc)

        # Compute the targets for each version.
        if self._forgetting_version == IncForgSPESNCell.FORGETTING_VERSION1:
            # Targets : what cannot be predicted by the
            # current matrix D.
            # Y = Win * U - D * X(t-1)
            Y = (torch.mm(self.w_in, U.t()) - torch.mm(self.D, X_old.t())).t()
        elif self._forgetting_version == IncForgSPESNCell.FORGETTING_VERSION2:
            # Targets : what cannot be predicted by the space
            # of D in conflict with the current patterns.
            # (using (A and C)).
            # Y = Win * U - D * E * X(t-1)
            Y = (torch.mm(self.w_in, U.t()) - torch.mm(torch.mm(self.D, E.C), X_old.t())).t()
        elif self._forgetting_version == IncForgSPESNCell.FORGETTING_VERSION3:
            # Targets : what cannot be predicted by the space
            # of D in conflict with the current patterns.
            # (using C instead of (A and C)
            # Y = Win * U - C * D * X(t-1)
            Y = (torch.mm(self.w_in, U.t()) - torch.mm(torch.mm(C.C, self.D), X_old.t())).t()
        elif self._forgetting_version == IncForgSPESNCell.FORGETTING_VERSION4:
            # Targets : what cannot be predicted by the current matrix D.
            # Y = Win * U - D * X(t-1)
            Y = (torch.mm(self.w_in, U.t()) - torch.mm(self.D, X_old.t())).t()
        elif self._forgetting_version == IncForgSPESNCell.FORGETTING_VERSION5:
            # Targets : what cannot be predicted by the current matrix D and (!)
            # the added increment matrix Dinc.
            # Y = Win * U - (D + Dinc) * X(t-1)
            Y = (torch.mm(self.w_in, U.t()) - torch.mm(self.D + self.Dinc, X_old.t())).t()
        # end if

        # Debug
        self._call_debug_point("Td{}".format(self._n_samples), Y, "IncSPESNCell", "_update_D_loading")

        # Compute the final matrix for the different version
        if self._forgetting_version == IncForgSPESNCell.FORGETTING_VERSION1:
            # Compute the increment and update for matrix D
            self.Dup = self._compute_update(X_old, Y, E, C)

            # Debug
            self._call_debug_point("Dinc{}".format(self._n_samples), self.Dinc, "IncSPESNCell", "_update_D_loading")
            self._call_debug_point("Dup{}".format(self._n_samples), self.Dup, "IncSPESNCell", "_update_D_loading")

            # Compute final matrix
            if self._conceptors.A().quota + C.quota > self._forgetting_threshold:
                # D = (1-l) * D + l * Dup + Dinc
                self.D = (1.0 - self._lambda) * self.D + self._lambda * self.Dup + self.Dinc
            else:
                # D = D + Dinc
                self.D += self.Dinc
            # end if
        elif self._forgetting_version == IncForgSPESNCell.FORGETTING_VERSION2:
            # Compute the increment and update for matrix D
            self.Dup = self._compute_update(X_old, Y, E, C)

            # Debug
            self._call_debug_point("Dinc{}".format(self._n_samples), self.Dinc, "IncSPESNCell", "_update_D_loading")
            self._call_debug_point("Dup{}".format(self._n_samples), self.Dup, "IncSPESNCell", "_update_D_loading")

            # Compute final matrix
            if self._conceptors.A().quota + C.quota > self._forgetting_threshold:
                # D = M * D + (1-l) * E * D + l * Dup + Dinc
                self.D = torch.mm(M.C, self.D) + (1.0 - self._lambda) * torch.mm(E.C, self.D) + self._lambda * self.Dup + self.Dinc
            else:
                # D = M * D + E * D + Dinc
                self.D += torch.mm(M.C, self.D) + torch.mm(E.C, self.D) + self.Dinc
            # end if
        elif self._forgetting_version == IncForgSPESNCell.FORGETTING_VERSION3:
            # Compute the increment and update for matrix D
            self.Dup = self._compute_update(X_old, Y, E, C)

            # Debug
            self._call_debug_point("Dinc{}".format(self._n_samples), self.Dinc, "IncSPESNCell", "_update_D_loading")
            self._call_debug_point("Dup{}".format(self._n_samples), self.Dup, "IncSPESNCell", "_update_D_loading")

            # Compute final matrix
            if self._conceptors.A().quota + C.quota > self._forgetting_threshold:
                # D = -C * D + (1-l) * C * D + l * Dup + Dinc
                self.D = torch.mm(NC.C, self.D) + (1.0 - self._lambda) * torch.mm(C.C, self.D) + self._lambda * self.Dup + self.Dinc
            else:
                # D = -C * D + C * D + Dinc
                self.D += torch.mm(NC.C, self.D) + torch.mm(C.C, self.D) + self.Dinc
            # end if
        elif self._forgetting_version == IncForgSPESNCell.FORGETTING_VERSION4:
            # Compute new D to predict the new pattern
            self.Dnew = self._compute_update(X_old, Y, E, C)

            # Compute the increment and update for matrix D
            self.Dinc = torch.mm(self._conceptors.F(), self.Dnew)
            self.Dup = torch.mm(self._conceptors.A().C, self.Dnew)

            # Debug
            self._call_debug_point("Dnew{}".format(self._n_samples), self.Dnew, "IncSPESNCell", "_update_D_loading")
            self._call_debug_point("Dinc{}".format(self._n_samples), self.Dinc, "IncSPESNCell", "_update_D_loading")
            self._call_debug_point("Dup{}".format(self._n_samples), self.Dup, "IncSPESNCell", "_update_D_loading")

            # Compute final matrix
            if self._conceptors.A().quota + C.quota > self._forgetting_threshold:
                # D = -C * D + (1-l) * C * D + l * Dup + Dinc
                self.D = torch.mm(NC.C, self.D) + (1.0 - self._lambda) * torch.mm(C.C, self.D) + self._lambda * self.Dup + self.Dinc
            else:
                # D = D + -C * D + C * D + Dinc
                self.D += torch.mm(NC.C, self.D) + torch.mm(C.C, self.D) + self.Dinc
            # end if
        elif self._forgetting_version == IncForgSPESNCell.FORGETTING_VERSION5:
            # Compute matrix Dnew which predict what cannot be predicted by D + Dinc
            self.Dnew = self._compute_update(X_old, Y, E, C)

            # Split the matrix Dnew into update and increment
            # Dup = A * Dnew
            self.Dup = torch.mm(self._conceptors.A().C, self.Dnew)

            # Debug
            self._call_debug_point("Dnew{}".format(self._n_samples), self.Dnew, "IncSPESNCell", "_update_D_loading")
            self._call_debug_point("Dinc{}".format(self._n_samples), self.Dinc, "IncSPESNCell", "_update_D_loading")
            self._call_debug_point("Dup{}".format(self._n_samples), self.Dup, "IncSPESNCell", "_update_D_loading")

            # Compute final matrix
            if self._conceptors.A().quota + C.quota > self._forgetting_threshold:
                # D = -C * D + (1-l) * C * D + l * Dup + Dinc
                self.D = torch.mm(NC.C, self.D) + (1.0 - self._lambda) * torch.mm(C.C, self.D) + self._lambda * self.Dup + self.Dinc
            else:
                # D = -C * D + C * D + Dinc
                self.D += torch.mm(NC.C, self.D) + torch.mm(C.C, self.D) + self.Dinc
            # end if
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

        # Compute conflict zone
        C, NC = self._compute_conceptor(X)

        # Targets : what cannot be predicted by the
        # current matrix D.
        Y = (U.t() - torch.mm(self.R, X_old.t())).t()
        self._call_debug_point("Td{}".format(self._n_samples), Y, "IncSPESNCell", "_update_R_loading")

        # Compute the increment for matrix D
        self.Rinc = self._compute_increment(X_old, Y)

        # Compute the update for matrix R
        self.Rup = self._compute_update(X_old, Y)

        # Debug
        self._call_debug_point("Rinc{}".format(self._n_samples), self.Rinc, "IncSPESNCell", "_update_R_loading")

        # Increment D
        if self._forgetting_version == IncForgSPESNCell.FORGETTING_A:
            if self._conceptors.A().quota + C.quota > self._forgetting_threshold:
                self.R = (1.0 - self._lambda) * self.R + self._lambda * self.Rup + self.Rinc
            else:
                self.R += self.Rinc
            # end if
        else:
            if self._conceptors.A().quota + C.quota > self._forgetting_threshold:
                self.R = torch.mm(NC.C, self.R.t()).t() + (1.0 - self._lambda) * torch.mm(C.C, self.R.t()).t() + self._lambda * self.Rup + self.Rinc
            else:
                self.R = torch.mm(NC.C, self.R.t()).t() + torch.mm(C.C, self.R.t()).t() + self.Rinc
            # end if
        # end if

        # Debug
        self._call_debug_point("R{}".format(self._n_samples), self.R, "IncSPESNCell", "_update_R_loading")
    # end _update_R_loading

    # endregion PRIVATE

    # region OVERRIDE

    # endregion OVERRIDE

# end IncSPESNCell
