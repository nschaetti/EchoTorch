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
from .IncSPESNCell import IncSPESNCell
from .Conceptor import Conceptor
from echotorch.utils import nrmse
from echotorch.utils import quota, rank


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
    def __init__(self, ridge_param_inc=0.0, ridge_param_up=0.0, lambda_param=0.0, forgetting_threshold=0.95,
                 forgetting_version=FORGETTING_VERSION1, *args, **kwargs):
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
        self._ridge_param_inc = ridge_param_inc
        self._ridge_param_up = ridge_param_up
        self._lambda = lambda_param
        self._forgetting_version = forgetting_version
        self._forgetting_threshold = forgetting_threshold

        # Debug info
        self.dinc_nrmse = -1
        self.dup_nrmse = -1
        self.dinc_magnitude = 0
        self.dup_magnitude = 0
        self.d_magnitude = 0
        self.d_gradient = 0
        self.d_rank = 0
        self.dinc_rank = 0
        self.dup_rank = 0
        self.e_rank = 0
        self.d_SVs = None
        self.dinc_SVs = None
        self.dup_SVs = None
        self.e_SVs = None

        # Spaces used for loading
        self.C = None
        self.A = Conceptor.empty(self.output_dim, dtype=self._dtype)
        self.E = None
        self.M = None

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

    # region PUBLIC

    # Set aperture
    def set_aperture(self, aperture):
        """
        Set aperture
        """
        self._aperture = aperture
    # end set_aperture

    # endregion PUBLIC

    # region PRIVATE

    # Compute update matrix (Dup)
    def _compute_update(self, X_old, Y, E, ridge_param):
        """
        Compute update matrix
        :param X_old: Reservoir states at time t - 1
        :param Y: Targets to predict (inputs)
        :return: Matrix update Dup
        """
        # Learn length
        learn_length = X_old.size(0)

        # If the conflict zone is not empty, we compute
        # the features as the reservoir states restricted
        # to the conflict zone.
        if not E.is_null():
            S_old = torch.mm(E.C, X_old.t()).t()
        else:
            S_old = X_old
        # end if

        # Debug
        self._call_debug_point("S_old{}".format(self._n_samples), S_old, "IncForgSPESNCell", "_compute_update")

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
        ridge_sTs = sTs + ridge_param * torch.eye(self._output_dim)

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

        # Compute the update for matrix D
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

        # SV modification function
        def modify_SVs(svs):
            svs[svs > 0.5] = 1.0
            svs[svs <= 0.5] = 0.0
            return svs
        # end modify_SVs

        # SV modification with tanh
        def modify_SVs_tanh(svs):
            for i in range(svs.size(0)):
                # svs[i] = (torch.tanh(50.0 * (2.0 * svs[i] - 1))) / 2.0
                svs[i] = (torch.tanh(svs[i] * 50 - 25) + 1) / 2.0
            # end for
            return svs
        # end modify_SVs_tanh

        # Modify conceptor's SVs
        C.modify_SVs(modify_SVs)

        return C
    # end if

    # Compute A (space occupied by all patterns)
    def _update_A(self, M, C, increment=False):
        """
        Compute A (space occupied by all patterns
        :param C: Conceptor
        """
        if increment:
            self.A = Conceptor.operator_OR(self.A, C)
        else:
            self.A = Conceptor.operator_OR(M, C)
        # end if
    # end _update_A

    # Compute F (space not occupied by any pattern)
    def _compute_F_matrix(self):
        """
        Compute F (space not occupied by any pattern)
        """
        if not self.A.is_null():
            return Conceptor.operator_NOT(self.A).C
        else:
            return Conceptor.identity(self.output_dim, dtype=self._dtype).C
        # end if
    # end _compute_F

    # Compute M (conflict free zone)
    def _compute_M(self, C):
        """
        Compute M (conflict free zone)
        :param C: Conceptor of current pattern.
        """
        return Conceptor.operator_AND(self.A, Conceptor.operator_NOT(C))
    # end _compute_M

    # Compute E (conflict zone)
    def _compute_E(self, M):
        """
        Compute E (conflict zone)
        :param M: conflict free zone.
        """
        if not M.is_null():
            return Conceptor.operator_NOT(M)
        else:
            return Conceptor.identity(self.output_dim, dtype=self._dtype)
        # end if
    # end _compute_E

    # Update input simulation matrix D
    def _update_D_loading(self, states, inputs):
        """
        Update input simulation matrix D
        """
        # Get X and U
        X, X_old, U = self._compute_XU(states, inputs)

        # Compute conceptor of current pattern
        self.C = self._compute_conceptor(X)

        # Compute free zone
        self.F = self._compute_F_matrix()

        # Compute demilitarized zone and conflict zone
        self.M = self._compute_M(self.C)
        self.E = self._compute_E(self.M)

        # DEBUG E
        self.e_rank = rank(self.E.C)
        _, self.e_SVs, _ = torch.svd(self.E.C)

        # Truth for comparison and test (Win * U)
        win_u = torch.mm(self.w_in, U.t()).t()

        # Targets for the increment matrix Dinc(t+1)
        # What cannot be learned by D(t) alone.
        Yinc = (torch.mm(self.w_in, U.t()) - torch.mm(self.D, X_old.t())).t()

        # Compute the increment for matrix D
        # with adaptive ridge param (*1000 if free zone F is too small)
        if quota(self.F) < 1e-1:
            self.Dinc = self._compute_increment(X_old, Yinc, self._ridge_param_inc * 1000, F=self._compute_F_matrix())
        else:
            self.Dinc = self._compute_increment(X_old, Yinc, self._ridge_param_inc, F=self._compute_F_matrix())
        # end if

        # DEBUG Dinc
        self.dinc_nrmse = nrmse(torch.mm(self.D + self.Dinc, X_old.t()).t(), win_u)
        self.dinc_magnitude = torch.norm(self.Dinc)
        self.dinc_rank = rank(self.Dinc)
        _, self.dinc_SVs, _ = torch.svd(self.Dinc)

        # Targets for the update matrix Dup(t+1)
        # What cannot be predicted by the current matrix D(t)
        # Y = Win * U - D * X(t-1)
        Y = (torch.mm(self.w_in, U.t()) - torch.mm(self.D, X_old.t())).t()
        # Y = (torch.mm(self.w_in, U.t()) - torch.mm(self.D, torch.mm(self.M.C, X_old.t()))).t()
        # Y = (torch.mm(self.w_in, U.t())).t()

        # Debug
        self._call_debug_point("Td{}".format(self._n_samples), Y, "IncSPESNCell", "_update_D_loading")

        # Compute matrix Dup which predict what cannot be predicted by D using the conflict zone E and
        # free zone F (E+F)
        self.Dup = self._compute_update(X_old, Y, self.E, self._ridge_param_up)

        # DEBUG Dup
        self.dup_nrmse = nrmse(torch.mm(self.D + self.Dup, X_old.t()).t(), win_u)
        # self.dup_nrmse = nrmse(torch.mm(torch.mm(self.M.C, self.D) + (1.0 - self._lambda) * torch.mm(self.E.C, self.D) + self._lambda * self.Dup, X_old.t()).t(), win_u)
        self.dup_magnitude = torch.norm(self.Dup)
        self.dup_rank = rank(self.Dup)
        _, self.dup_SVs, _ = torch.svd(self.Dup)

        # Debug
        self._call_debug_point("Dinc{}".format(self._n_samples), self.Dinc, "IncSPESNCell", "_update_D_loading")
        self._call_debug_point("Dup{}".format(self._n_samples), self.Dup, "IncSPESNCell", "_update_D_loading")

        # We erase information in D with Dup if above a threshold
        if self.A.quota > self._forgetting_threshold:
            # Gradient
            self.d_gradient = torch.norm(self.D + self._lambda * self.Dup) - torch.norm(self.D)
            # self.d_gradient = torch.norm(torch.mm(self.M.C, self.D) + (1.0 - self._lambda) * torch.mm(self.E.C, self.D) + self._lambda * self.Dup) - torch.norm(self.D)
            print("update")
            # D = D + Dup
            self.D += self._lambda * self.Dup
            # self.D = torch.mm(self.M.C, self.D) + (1.0 - self._lambda) * torch.mm(self.E.C, self.D) + self._lambda * self.Dup

            # Update A
            self._update_A(self.M, self.C)
        else:
            # Gradient
            self.d_gradient = torch.norm(self.D + self.Dinc) - torch.norm(self.D)

            # D = D + Dinc
            self.D += self.Dinc
            # self.D += self._lambda * self.Dup

            # Update A
            self._update_A(self.M, self.C, increment=True)
            # self._update_A(self.M, C)
        # end if

        # Save D's magnitude and rank and SV
        self.d_magnitude = torch.norm(self.D)
        self.d_rank = rank(self.D)
        _, self.d_SVs, _ = torch.svd(self.D)

        # Debug
        self._call_debug_point("D{}".format(self._n_samples), self.D, "IncSPESNCell", "_update_D_loading")
    # end _update_D_loading

    # Update input recreation matrix R
    """def _update_R_loading(self, states, inputs):
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
    # end _update_R_loading"""

    # endregion PRIVATE

    # region OVERRIDE

    # endregion OVERRIDE

# end IncSPESNCell
