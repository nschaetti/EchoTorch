# -*- coding: utf-8 -*-
#
# File : echotorch/nn/utils/JoinStates.py
# Description : A layer joining states over time.
# Date : 6th of November, 2019
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


import torch
import torch.nn as nn
from torch.autograd import Variable


# Join States Layer
class JoinStates(nn.Module):
    """
    Join States Layer
    """

    # Constructor
    def __init__(self, hidden_dim, join_length):
        """
        Constructor
        """
        # Super
        super(JoinStates, self).__init__()

        # Length and hidden dim
        self._hidden_dim = hidden_dim
        self._join_length = join_length
    # end __init__

    #################
    # PUBLIC
    #################

    # Forward
    def forward(self, x):
        """
        Forward
        :return: Module's output
        """
        # Batch size
        batch_size = x.size(0)

        # Time length
        time_length = x.size(1)

        # New time length
        new_time_length = int(time_length / self._join_length)

        return x.reshape(batch_size, new_time_length, self._hidden_dim * self._join_length)
    # end forward

# end JoinStates
