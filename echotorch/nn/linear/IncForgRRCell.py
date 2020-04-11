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
from echotorch.utils import quota, rank


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
    def __init__(self, aperture, ridge_param_inc=0.01, ridge_param_up=0.01, lambda_param=0.0, forgetting_threshold=0.95,
                 forgetting_version=FORGETTING_VERSION1, *args, **kwargs):
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
        self._ridge_param_inc = ridge_param_inc
        self._ridge_param_up = ridge_param_up
        self._aperture = aperture
        self._lambda = lambda_param
        self._forgetting_version = forgetting_version
        self._forgetting_threshold = forgetting_threshold

        # Debug
        self.w_out_inc_nrmse = -1
        self.w_out_up_nrmse = -1
        self.w_out_inc_magnitude = 0
        self.w_out_up_magnitude = 0
        self.w_out_magnitude = 0
        self.w_out_gradient = 0
        self.w_out_inc_rank = 0
        self.w_out_up_rank = 0
        self.w_out_rank = 0
        self.e_rank = 0
        self.w_out_SVs = None
        self.w_out_inc_SVs = None
        self.w_out_up_SVs = None

        # Space used by all patterns
        self.C = None
        self.A = Conceptor.empty(self.output_dim)

        # Wout matrix, update, increment and new
        self.register_buffer('w_out_up', Variable(torch.zeros(1, self.input_dim, dtype=self.dtype), requires_grad=False))
        self.register_buffer('w_out_new', Variable(torch.zeros(1, self.input_dim, dtype=self.dtype), requires_grad=False))
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

    # Compute update matrix for Wout
    def _compute_update(self, X, Y, E, ridge_param):
        """
        Compute update matrix for Wout
        :param X:
        :param Y:
        :param E:
        :param C:
        """
        # Time length
        time_length = X.size()[0]

        # If the conflict + free zone is not empty
        # then we predict using states restricted to that zones.
        if not E.is_null():
            # S = E * X
            S = torch.mm(E.C, X.t()).t()
        else:
            # No filtering
            S = X.t()
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

        # SV modification function
        def modify_SVs(svs):
            svs[svs > 0.5] = 1.0
            svs[svs <= 0.5] = 0.0
            return svs
        # end if

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

    # Update Wout matrix
    def _update_Wout_loading(self, X, Y):
        """
        Update Wout matrix
        """
        # Compute zones
        self.C = self._compute_conceptor(X)

        # Compute free zone
        self.F = self._compute_F_matrix()

        # Compute conflict and conflict free zones
        self.M = self._compute_M(self.C)
        self.E = self._compute_E(self.M)

        # Debug
        self.e_rank = rank(self.E.C)

        # Compute increment for Wout
        if quota(self.F) < 1e-1:
            self.w_out_inc = self._compute_increment(X, Y, self._ridge_param_inc * 1000, F=self._compute_F_matrix())
        else:
            self.w_out_inc = self._compute_increment(X, Y, self._ridge_param_inc, F=self._compute_F_matrix())
        # end if

        # Debug
        self._call_debug_point("w_out_inc{}".format(self._n_samples), self.w_out_inc, "IncForgRRCell", "_update_Wout_loading")

        # DEBUG Wout inc
        self.w_out_inc_nrmse = nrmse(torch.mm(self.w_out + self.w_out_inc, X.t()).t(), Y)
        self.w_out_inc_magnitude = torch.norm(self.w_out_inc)
        self.w_out_inc_rank = rank(self.w_out_inc)
        _, self.w_out_inc_SVs, _ = torch.svd(self.w_out_inc)

        # Targets: what cannot be predicted by the current matrix Wout.
        # Yt = Y - Wout * X
        Yt = Y - torch.mm(self.w_out, X.t()).t()
        # Yt = Y - torch.mm(self.w_out, torch.mm(self.M.C, X.t())).t()

        # Debug
        self._call_debug_point("Td{}".format(self._n_samples), Y, "IncForgRRCell", "_update_Wout_loading")

        # Compute matrix Wout update which predict what cannot be predicted by Wout using the conflict zone
        # and the free zone.
        self.w_out_up = self._compute_update(X, Yt, self.E, self._ridge_param_up)

        # Split the matrix Wout_new into update and increment
        # Wout_up = A * Wout_new
        self.w_out_up_nrmse = nrmse(torch.mm(self.w_out + self.w_out_up, X.t()).t(), Y)
        self.w_out_up_magnitude = torch.norm(self.w_out_up)
        self.w_out_up_rank = rank(self.w_out_up)
        _, self.w_out_up_SVs, _ = torch.svd(self.w_out_up)

        # Debug
        self._call_debug_point("w_out_up{}".format(self._n_samples), self.w_out_up, "IncForgRRCell", "_update_Wout_loading")

        # Compute final matrix
        if self.A.quota > self._forgetting_threshold:
            # Gradient
            self.w_out_gradient = torch.norm(self.w_out + self.w_out_up) - torch.norm(self.w_out)

            # Wout = Wout + Wout_up
            self.w_out += self.w_out_up

            # Update A
            self._update_A(self.M, self.C)
        else:
            # Gradient
            self.w_out_gradient = torch.norm(self.w_out + self.w_out_inc) - torch.norm(self.w_out)

            # Wout = Wout + Wout_inc
            self.w_out += self.w_out_inc
            # self.w_out += self.w_out_up

            # Update A
            self._update_A(self.M, self.C, increment=True)
            # self._update_A(self.M, C)
        # end if

        # Wout magnitude
        self.w_out_magnitude = torch.norm(self.w_out)
        self.w_out_rank = rank(self.w_out)
        _, self.w_out_SVs, _ = torch.svd(self.w_out)

        # Debug
        self._call_debug_point("w_out{}".format(self._n_samples), self.w_out, "IncForgRRCell", "_update_Wout_loading")
    # end _update_Wout_loading

    # endregion OVERRIDE

# end IncForgRRCell
