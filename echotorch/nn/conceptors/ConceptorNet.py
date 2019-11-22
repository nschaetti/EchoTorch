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
import torch
from ..reservoir import ESN


# Conceptor Network
class ConceptorNet(ESN):
    """
    Conceptor Network
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, output_dim, esn_cell, dtype=torch.float32):
        """
        Constructor
        :param input_dim: Input feature space dimension
        :param hidden_dim: Hidden space dimension
        :param output_dim: Output space dimension
        :param esn_cell: ESN cell
        :param n_conceptors: Number of conceptor to create.
        :param dtype: Data type
        """
        super(ConceptorNet, self).__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            w_generator=None,
            win_generator=None,
            wbias_generator=None,
            washout=esn_cell.washout,
            create_rnn=False,
            create_output=True,
            dtype=dtype
        )

        # Properties
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._conceptor_active = False
        self._dtype = dtype

        # Current conceptor
        self._current_conceptor = None

        # Recurrent layer
        self._esn_cell = esn_cell

        # Neural filter
        self._esn_cell.connect("neural-filter", self._neural_filter)
        self._esn_cell.connect("post-states-update", self._post_update_states)

        # Forward hook to learn conceptor
        # self._esn_cell.register_forward_hook(self._forward_hook_conceptor_learning)

        # Trainable elements
        self.add_trainable(self._esn_cell)
    # end __init__

    ####################
    # PUBLIC
    ####################

    # Set the current conceptor
    def set_conceptor(self, C):
        """
        Set the current conceptor
        :param C: The conceptor matrix
        """
        self._current_conceptor = C
    # end set_conceptor

    # Use conceptor ?
    def conceptor_active(self, value):
        """
        Use conceptors ?
        :param value: True/False
        """
        self._conceptor_active = value
    # end conceptor_active

    ####################
    # PRIVATE
    ####################

    # Neural filter / training
    def _neural_filter(self, x, ut, t, washout):
        """
        Neural filter
        :param x: States to filter
        :param ut: Inputs
        :param t: Time t
        :param washout: In washout period
        """
        if self._conceptor_active and self._current_conceptor is not None and not self._current_conceptor.training:
            return self._current_conceptor(x)
        else:
            return x
        # end if
    # end _neural_filter

    # Get states after batch update to train conceptors
    def _post_update_states(self, states, inputs, b):
        """
        Get states after batch update to train conceptors
        :param states: Reservoir states (without washout)
        :param inputs: Input signal
        :param b: Batch position
        """
        if self._conceptor_active and self._current_conceptor is not None and self._current_conceptor.training:
            self._current_conceptor(states)
        # end if
    # end _post_update_states

    # Hook executed to learn conceptors
    def _forward_hook_conceptor_learning(self, module, inputs, outputs):
        """
        Hook executed to learn conceptors
        :param module: Module hooked
        :param inputs: Module's inputs
        :param outputs: Module's outputs
        """
        self._current_conceptor(outputs)
    # end _forward_hook_conceptor_learning

# end ConceptorNet
