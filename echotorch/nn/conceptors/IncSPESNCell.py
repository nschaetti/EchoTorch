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

        # Debug
        self._call_debug_point("D{}".format(self._n_samples), self.D, "IncSPESNCell", "finalize")

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
            return self.D.mv(self.hidden)
        else:
            return self.w_in.mv(ut)
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
            # X (reservoir states)
            X = states[self._washout:]
            self._call_debug_point("X{}".format(self._n_samples), X, "SPESNCell", "_post_update_hook")

            # Inputs
            U = inputs[self._washout:]

            # Get old states as features
            if self._fill_left:
                X_old = self.features(X, fill_left=states[self._washout-1] if self._washout > 0 else None)
            else:
                X_old = self.features(X)
            # end if

            # Debug Xold
            self._call_debug_point("Xold{}".format(self._n_samples), X_old, "IncSPESNCell", "_post_update_hook")

            # Learn length
            learn_length = X_old.size(0)

            # Targets : what cannot be predicted by the
            # current matrix D.
            Td = (torch.mm(self.w_in, U.t()) - torch.mm(self.D, X_old.t())).t()
            self._call_debug_point("Td{}".format(self._n_samples), Td, "IncSPESNCell", "_post_update_hook")

            # We filter X- if the conceptor is not null
            if not self._conceptors.is_null():
                # The linear subspace of the reservoir state space that are not yet
                # occupied by any pattern.
                F = self._conceptors.F()
                self._call_debug_point("F{}".format(self._n_samples), F, "IncSPESNCell", "_post_update_hook")

                # Filter old state to get only what is new
                S_old = torch.mm(F, X_old.t()).t()
            else:
                # No filter
                S_old = X_old
            # end if

            # Debug
            self._call_debug_point("Sold{}".format(self._n_samples), S_old, "IncSPESNCell", "_post_update_hook")

            # Targets
            if self._averaged:
                sTd = torch.mm(S_old.t(), Td) / learn_length
            else:
                sTd = torch.mm(S_old.t(), Td)
            # end if

            # Debug
            self._call_debug_point("sTd{}".format(self._n_samples), sTd, "IncSPESNCell", "_post_update_hook")

            # sTs
            if self._averaged:
                sTs = torch.mm(S_old.t(), S_old) / learn_length
            else:
                sTs = torch.mm(S_old.t(), S_old)
            # end if

            # Debug
            self._call_debug_point("sTs{}".format(self._n_samples), sTs, "IncSPESNCell", "_post_update_hook")

            # Ridge sTs
            ridge_sTs = sTs + math.pow(self._aperture, -2) * torch.eye(self._output_dim)

            # Debug
            self._call_debug_point("ridge_sTs{}".format(self._n_samples), ridge_sTs, "IncSPESNCell", "_post_update_hook")

            # Inverse / pinverse
            if self._w_learning_algo == "inv":
                inv_sTs = self._inverse("ridge_sTs", ridge_sTs, "IncSPESNCell", "_post_update_hook")
            elif self._w_learning_algo == "pinv":
                inv_sTs = self._pinverse("ridge_sTs", ridge_sTs, "IncSPESNCell", "_post_update_hook")
            else:
                raise Exception("Unknown learning method {}".format(self._learning_algo))
            # end if

            # Debug
            self._call_debug_point("inv_sTs{}".format(self._n_samples), inv_sTs, "IncSPESNCell", "_post_update_hook")

            # Compute the increment for matrix D
            Dinc = torch.mm(inv_sTs, sTd).t()

            # Debug
            self._call_debug_point("Dinc{}".format(self._n_samples), Dinc, "IncSPESNCell", "_post_update_hook")

            # Increment D
            self.D += Dinc

            # Debug
            self._call_debug_point("D{}".format(self._n_samples), self.D, "IncSPESNCell", "_post_update_hook")

            # Inc
            self._n_samples += 1
        # end if

        return states
    # end _post_update_hook

    # endregion OVERRIDE

# end IncSPESNCell
