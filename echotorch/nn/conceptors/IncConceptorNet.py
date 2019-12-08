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
from .. import Node
from ..linear import IncRRCell
from ..reservoir import ESN
from .ConceptorNet import ConceptorNet


# Incremental learning-based Conceptor Network
class IncConceptorNet(ConceptorNet):
    """
    Incremental learning-based Conceptor Network
    """
    # region BODY1

    # Constructor
    def __init__(self, input_dim, hidden_dim, output_dim, esn_cell, conceptors, ridge_param, learning_algo='inv',
                 softmax_output=False, debug=Node.NO_DEBUG, test_case=None, dtype=torch.float32):
        """
        Constructor
        :param input_dim: Input feature space dimension
        :param hidden_dim: Hidden space dimension
        :param output_dim: Output space dimension
        :param esn_cell: ESN cell
        :param conceptor: Neural filter
        :param dtype: Data type
        """
        super(IncConceptorNet, self).__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            esn_cell=esn_cell,
            conceptor=conceptors,
            create_output=False,
            test_case=test_case,
            dtype=dtype
        )

        # Output layer
        self._output = IncRRCell(
            input_dim=hidden_dim,
            output_dim=output_dim,
            ridge_param=ridge_param,
            learning_algo=learning_algo,
            softmax_output=softmax_output,
            conceptors=conceptors,
            debug=debug,
            test_case=test_case,
            dtype=dtype
        )
        self.add_trainable(self._output)
    # end __init__

    # region PUBLIC

    # endregion PUBLIC

    # region PRIVATE

    # endregion PRIVATE

    # region OVERRIDE

    # Extra-information
    def extra_repr(self):
        """
        Extra-information
        :return: String
        """
        s = super(IncConceptorNet, self).extra_repr()
        return s.format(**self.__dict__)
    # end extra_repr

    # endregion OVERRIDE

    # endregion BODY
# end IncConceptorNet
