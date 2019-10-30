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
    def __init__(self, input_dim, hidden_dim, output_dim, esn_cell, n_conceptors, dtype=torch.float32):
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
            create_rnn=False,
            create_output=True
        )

        # Properties
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._n_conceptor = n_conceptors
        self._dtype = dtype

        # Recurrent layer
        self._esn_cell = esn_cell

        # Neural filter
        self._esn_cell.connect("neural-filter", self._neural_filter)
    # end __init__

    ####################
    # PUBLIC
    ####################

    # Finish training
    def finalize(self):
        """
        Finish training
        """
        # Finalize internal training
        self._esn_cell.finalize()

        # Finalize output training
        self._output.finalize()

        # Not in training mode anymore
        self.train(False)
    # end finalize

    ####################
    # PRIVATE
    ####################

    # Neural filter / training
    def _neural_filter(self, x, ut, t):
        """
        Neural filter
        :param x:
        :param ut:
        :param t:
        :return:
        """
        print(t)
    # end _neural_filter

# end ConceptorNet
