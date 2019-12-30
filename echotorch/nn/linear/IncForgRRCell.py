# -*- coding: utf-8 -*-
#
# File : echotorch/nn/linear/IncRRCell.py
# Description : Incremental Ridge Regression node
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
Created on 5 November 2019
@author: Nils Schaetti
"""

# Imports
import torch.sparse
import torch
from torch.autograd import Variable
from ..conceptors import Conceptor
from .IncRRCell import IncRRCell
from echotorch.utils import nrmse


# Incremental Ridge Regression node
class IncForgRRCell(IncRRCell):
    """
    Incremental Ridge Regression node
    """

    # Forgetting version
    FORGETTING_VERSION1 = 1
    FORGETTING_VERSION2 = 2
    FORGETTING_VERSION3 = 3
    FORGETTING_VERSION4 = 4
    FORGETTING_VERSION5 = 5

    # Constructor
    def __init__(self, aperture, lambda_param=0.0, forgetting_threshold=0.95, forgetting_version=FORGETTING_VERSION1,
                 *args, **kwargs):
        """
        Constructor
        :param input_dim: Feature space dimension
        :param output_dim: Output space dimension
        :param conceptors: ConceptorSet object of conceptors used to describe space.
        :param ridge_param: Ridge parameter
        :param with_bias: Add a bias to the linear layer
        :param learning_algo: Inverse (inv) or pseudo-inverse (pinv)
        :param softmax_output: Add a softmax output (normalize outputs) ?
        :param averaged: Covariance matrix divided by the number of samples ?
        :param debug: Debug mode
        :param test_case: Test case to call for test.
        :param dtype: Data type
        """
        # Superclass
        super(IncForgRRCell, self).__init__(
            *args,
            **kwargs
        )

        # Properties
        self._aperture = aperture
        self._lambda = lambda_param
        self._forgetting_version = forgetting_version
        self._forgetting_threshold = forgetting_threshold
        self.w_out_inc_nrmse = -1
        self.w_out_up_nrmse = -1
        self.w_out_inc_magnitude = 0
        self.w_out_up_magnitude = 0
        self.w_out_magnitude = 0
        self.w_out_gradient = 0

        # Wout matrix, update, increment and new
        self.register_buffer('w_out_up', Variable(torch.zeros(1, self.input_dim, dtype=self.dtype), requires_grad=False))
        self.register_buffer('w_out_new', Variable(torch.zeros(1, self.input_dim, dtype=self.dtype), requires_grad=False))
    # end __init__

    # region PRIVATE

    # Compute update matrix for Wout
    def _compute_update(self, X, Y, E, C, NOT_M, ridge_param):
        """
        Compute update matrix for Wout
        :param X:
        :param Y:
        :param E:
        :param C:
        """
        # Time length
        time_length = X.size()[0]

        if self._forgetting_version == IncForgRRCell.FORGETTING_VERSION4:
            # X
            S = X
        # Filter if the conceptor is not null
        elif not self._conceptors.is_null():
            # The linear subspace of the reservoir state space
            # occupied by loaded patterns.
            A = self._conceptors.A().C

            # Debug
            self._call_debug_point("A{}".format(self._n_samples), A, "IncForgRRCell", "_compute_update")

            # For each version
            if self._forgetting_version == IncForgRRCell.FORGETTING_VERSION1:
                # Filter old state to get only what is new in that space
                # A * X
                S = torch.mm(E.C, X.t()).t()
            elif self._forgetting_version == IncForgRRCell.FORGETTING_VERSION2:
                # E * X
                S = torch.mm(E.C, X.t()).t()
            elif self._forgetting_version == IncForgRRCell.FORGETTING_VERSION3:
                # A * X
                S = torch.mm(A, X.t()).t()
            elif self._forgetting_version == IncForgRRCell.FORGETTING_VERSION5:
                # A * X
                S = torch.mm(E.C, X.t()).t()
            # end if
        else:
            # No update
            return torch.zeros(self.w_out.size(0), self.w_out.size(1), dtype=self.dtype)
        # end if

        # Debug
        self._call_debug_point("S{}".format(self._n_samples), S, "IncForgRRCell", "_compute_update")

        # sTs
        if self._averaged:
            sTs = torch.mm(S.t(), S) / time_length
        else:
            sTs = torch.mm(S.t(), S)
        # end if

        # Debug
        self._call_debug_point("sTs{}".format(self._n_samples), sTs, "IncForgRRCell", "_compute_update")

        # sTy
        if self._averaged:
            sTy = torch.mm(S.t(), Y) / time_length
        else:
            sTy = torch.mm(S.t(), Y)
        # end if

        # Debug
        self._call_debug_point("sTy{}".format(self._n_samples), sTy, "IncForgRRCell", "_compute_update")

        # Ridge sTs
        ridge_sTs = sTs + ridge_param * torch.eye(self._input_dim)

        # Debug
        self._call_debug_point("ridge_sTs{}".format(self._n_samples), ridge_sTs, "IncForgRRCell", "_compute_update")

        # Inverse / pinverse
        if self._learning_algo == "inv":
            inv_sTs = self._inverse("ridge_sTs", ridge_sTs, "IncForgRRCell", "_compute_update")
        elif self._learning_algo == "pinv":
            inv_sTs = self._pinverse("ridge_sTs", ridge_sTs, "IncForgRRCell", "_compute_update")
        else:
            raise Exception("Unknown learning method {}".format(self._learning_algo))
        # end if

        # Debug
        self._call_debug_point("inv_sTs{}".format(self._n_samples), inv_sTs, "IncForgRRCell", "_compute_update")

        # Compute increment for Wout
        return (torch.mm(inv_sTs, sTy)).t()
    # end _compute_update

    # endregion PRIVATE

    # region OVERRIDE

    # Compute conceptor of the current pattern
    def _compute_conceptor(self, X):
        """
        Compute
        """
        # Compute Conceptor for new pattern
        C = Conceptor(
            input_dim=self.input_dim,
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
        if E.is_null():
            M = Conceptor.operator_AND(A, NOT_C)
            NOT_M = torch.eye(self.input_dim)
        else:
            M = Conceptor.operator_AND(A, Conceptor.operator_NOT(E))
            NOT_M = Conceptor.operator_NOT(M).C
        # end if

        return C, NOT_C, E, M, NOT_M
    # end if

    # Update Wout matrix
    def _update_Wout_loading(self, X, Y):
        """
        Update Wout matrix
        """
        # Compute zones
        C, NC, E, M, NOT_M = self._compute_conceptor(X)

        # Compute increment for Wout
        self.w_out_inc = self._compute_increment(X, Y)

        # Compute the targets for each version.
        if self._forgetting_version == IncForgRRCell.FORGETTING_VERSION1:
            # Targets: What cannot be predicted by the current matrix Wout
            # Y - Wout * X
            Yt = Y - torch.mm(self.w_out, X.t()).t()
        elif self._forgetting_version == IncForgRRCell.FORGETTING_VERSION2:
            # Targets: what cannot be predicted by the space
            # of Wout in conflict with the current patterns.
            # (using (A and C)).
            # Y - Wout * E * X
            Yt = Y - torch.mm(self.w_out, torch.mm(E.C, X.t())).t()
        elif self._forgetting_version == IncForgRRCell.FORGETTING_VERSION3:
            # Targets: what cannot be predicted by the space of Wout
            # in conflict with the current patterns.
            # (using C instead of (A and C)).
            # Y - C * Wout * X
            Yt = Y - torch.mm(torch.mm(C.C, self.w_out.t()).t(), X.t()).t()
        elif self._forgetting_version == IncForgRRCell.FORGETTING_VERSION4:
            # Targets: what cannot be predicted by the current matrix Wout
            # Y - Wout * X
            Yt = Y - torch.mm(self.w_out, X.t()).t()
        elif self._forgetting_version == IncForgRRCell.FORGETTING_VERSION5:
            # Targets: what cannot be predicted by the current matrix Wout and (!)
            # the added increment matrix Wout_inc.
            # Y - (Wout + Wout_inc) * X
            Yt = Y - torch.mm(self.w_out + self.w_out_inc, X.t()).t()

            # Compute NRMSE of (Wout + Wout_inc) and magnitude
            self.w_out_inc_nrmse = nrmse(torch.mm(self.w_out + self.w_out_inc, X.t()).t(), Y)
            self.w_out_inc_magnitude = torch.norm(self.w_out_inc)
        # end if

        # Debug
        self._call_debug_point("Td{}".format(self._n_samples), Y, "IncForgRRCell", "_update_Wout_loading")

        # Compute the final matrix for different version
        if self._forgetting_version == IncForgRRCell.FORGETTING_VERSION1:
            # Compute the increment and update for matrix Wout
            self.w_out_up = self._compute_update(X, Yt, E, C, NOT_M, self._ridge_param)

            # Debug
            self._call_debug_point("w_out_up{}".format(self._n_samples), self.w_out_up, "IncForgRRCell", "_update_Wout_loading")

            # Compute final matrix
            if self._conceptors.A().quota + C.quota > self._forgetting_threshold:
                # Wout = (1-l) * Wout + l * Wout_up + Wout_inc
                self.w_out += self.w_out_up
            else:
                # Wout = Wout + Wout_inc
                self.w_out += self.w_out_inc
            # end if
        elif self._forgetting_version == IncForgRRCell.FORGETTING_VERSION2:
            # Compute the increment and update for matrix Wout
            self.w_out_up = self._compute_update(X, Yt, E, C, NOT_M, self._ridge_param)

            # Debug
            self._call_debug_point("w_out_up{}".format(self._n_samples), self.w_out_up, "IncForgRRCell", "_update_Wout_loading")

            # Compute final matrix
            if self._conceptors.A().quota + C.quota > self._forgetting_threshold:
                # Wout = M * Wout + (1-l) * E * Wout + l * Wout_up + Wout_inc
                self.w_out = torch.mm(M.C, self.w_out.t()).t() + (1.0 - self._lambda) * torch.mm(E.C, self.w_out.t()).t() + self._lambda * self.w_out_up + self.w_out_inc
            else:

                # Wout = M * Wout + E * Wout + Wout_inc
                self.w_out = torch.mm(M.C, self.w_out.t()).t() + torch.mm(E.C, self.w_out.t()).t() + self.w_out_inc
            # end if
        elif self._forgetting_version == IncForgRRCell.FORGETTING_VERSION3:
            # Compute the increment and update for matrix Wout
            self.w_out_up = self._compute_update(X, Yt, E, C, NOT_M, self._ridge_param)

            # Debug
            self._call_debug_point("w_out_up{}".format(self._n_samples), self.w_out_up, "IncForgRRCell", "_update_Wout_loading")

            # Compute final matrix
            if self._conceptors.A().quota + C.quota > self._forgetting_threshold:
                # Wout = -C * Wout + (1-l) * C * Wout + l * Wout_up + Wout_inc
                self.w_out = torch.mm(NC.C, self.w_out.t()).t() + (1.0 - self._lambda) * torch.mm(C.C, self.w_out.t()).t() + self._lambda * self.w_out_up + self.w_out_inc
            else:
                # Wout = -C * Wout + C * Wout + Wout_inc
                self.w_out = torch.mm(NC.C, self.w_out.t()).t() + torch.mm(C.C, self.w_out.t()).t() + self.w_out_inc
            # end if
        elif self._forgetting_version == IncForgRRCell.FORGETTING_VERSION4:
            # Compute new Wout to predict the new pattern
            self.w_out_new = self._compute_update(X, Yt, E, C, NOT_M, self._ridge_param)

            # Split Wout new into update and increment
            # Wout_inc = F * Wout_new
            # Wout_up = A * Wout_new
            self.w_out_inc = torch.mm(self._conceptors.F().t(), self.w_out_new.t()).t()
            self.w_out_up = torch.mm(self._conceptors.A().C.t(), self.w_out_new.t()).t()

            # Debug
            self._call_debug_point("w_out_new{}".format(self._n_samples), self.w_out_new, "IncForgRRCell", "_update_Wout_loading")
            self._call_debug_point("w_out_inc{}".format(self._n_samples), self.w_out_inc, "IncForgRRCell", "_update_Wout_loading")
            self._call_debug_point("w_out_up{}".format(self._n_samples), self.w_out_up, "IncForgRRCell", "_update_Wout_loading")

            # Compute final matrix
            if self._conceptors.A().quota + C.quota > self._forgetting_threshold:
                # Wout = -C * Wout + (1-l) * C * Wout + l * Wout_up + Wout_inc
                self.w_out = torch.mm(NC.C, self.w_out.t()).t() + (1.0 - self._lambda) * torch.mm(C.C, self.w_out.t()).t() + self._lambda * self.w_out_up + self.w_out_inc
            else:
                # Wout = -C * Wout + C * Wout + Wout_inc
                self.w_out = torch.mm(NC.C, self.w_out.t()).t() + torch.mm(C.C, self.w_out.t()).t() + self.w_out_inc
            # end if
        elif self._forgetting_version == IncForgRRCell.FORGETTING_VERSION5:
            # Compute matrix Wout which predict what cannot be predicted by Wout + Wout_inc
            self.w_out_up = self._compute_update(X, Yt, E, C, NOT_M, self._ridge_param)

            # Split the matrix Wout_new into update and increment
            # Wout_up = A * Wout_new
            self.w_out_up_nrmse = nrmse(torch.mm(self.w_out + self.w_out_inc + self.w_out_up, X.t()).t(), Y)
            self.w_out_up_magnitude = torch.norm(self.w_out_up)

            # Debug
            # self._call_debug_point("w_out_new{}".format(self._n_samples), self.w_out_new, "IncForgRRCell", "_update_Wout_loading")
            self._call_debug_point("w_out_inc{}".format(self._n_samples), self.w_out_inc, "IncForgRRCell", "_update_Wout_loading")
            self._call_debug_point("w_out_up{}".format(self._n_samples), self.w_out_up, "IncForgRRCell", "_update_Wout_loading")

            # Compute final matrix
            if self._conceptors.A().quota + C.quota > self._forgetting_threshold:
                # Wout = -C * Wout + (1-l) * C * Wout + l * Wout_up + Wout_inc
                self.w_out += self.w_out_inc + self.w_out_up
            else:
                # Gradient
                self.w_out_gradient = torch.norm(self.w_out + self.w_out_inc) - torch.norm(self.w_out)
                # Wout = -C * Wout + C * Wout + Wout_inc
                # self.w_out = torch.mm(NC.C, self.w_out.t()).t() + torch.mm(C.C, self.w_out.t()).t() + self.w_out_inc
                # self.w_out += self.w_out_up + self.w_out_inc
                self.w_out += self.w_out_inc
            # end if
        # end if

        # Wout magnitude
        self.w_out_magnitude = torch.norm(self.w_out)
    # end _update_Wout_loading

    # endregion OVERRIDE

# end IncForgRRCell
