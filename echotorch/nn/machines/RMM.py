# -*- coding: utf-8 -*-
#
# File : echotorch/nn/machines/RMM.py
# Description : Reservoir Memory Machines (paassen2020)
# Date : 17th of July, 2020
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
Created on 17th of July, 2020
@author: Nils Schaetti
"""

# Imports
import torch
from echotorch.nn.linear.RRCell import RRCell
from ..reservoir import ESN
from ..Node import Node


# Reservoir Memory Machines
class RMM(ESN):
    """
    Reservoir Memory Machines
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, output_dim, memory_dim, w_generator, win_generator, wbias_generator,
                 input_scaling=1.0, nonlin_func=torch.tanh, learning_algo='inv', controller_initialization='identity',
                 permit_duplicates=False, input_normalization=True, ridge_param=0.00001, with_bias=True,
                 softmax_output=False, washout=0, debug=Node.NO_DEBUG, test_case=None, dtype=torch.float32):
        """
        Constructor
        """
        super(RMM, self).__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            w_generator=w_generator,
            win_generator=win_generator,
            wbias_generator=wbias_generator,
            input_scaling=input_scaling,
            nonlin_func=nonlin_func,
            learning_algo=learning_algo,
            ridge_param=ridge_param,
            with_bias=with_bias,
            softmax_output=softmax_output,
            create_rnn=True,
            create_output=True,
            washout=washout,
            input_normalization=input_normalization,
            debug=debug,
            test_case=test_case,
            dtype=dtype
        )

        # Properties
        self._memory_dim = memory_dim
        self._controller_initialization = controller_initialization
        self._permit_duplicates = permit_duplicates


    # end __init__

# end RMM
