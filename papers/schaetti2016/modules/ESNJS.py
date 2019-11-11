# -*- coding: utf-8 -*-
#
# File : papers/schaetti2016/transforms/Concat.py
# Description : Transform images to a concatenation of multiple transformations.
# Date : 11th of November, 2019
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


# Imports
import torch.nn as nn
import echotorch.nn.reservoir
import echotorch.nn.utils


# ESN with Join State
class ESNJS(nn.Module):
    """
    ESN with Join State
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Constructor
        """
        # Generators

        # Create ESM
        esn = echotorch.nn.reservoir.LiESNCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )

        # Join states
        js = echotorch.nn.utils.JoinStates(input_dim=hidden_dim)
    # end __init__

    # Forward
    def forward(self, u):
        """
        Forward
        :param u:
        :return:
        """

# end ESNJS
