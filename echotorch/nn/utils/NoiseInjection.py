# -*- coding: utf-8 -*-
#
# File : echotorch/nn/NoiseInjection.py
# Description : Insert noise into the reservoir.
# Date : 28th November 2019
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
from ..Node import Node
from torch.autograd import Variable


# Inject noise into reservoir
# states.
class NoiseInjection(Node):
    """
    Inject noise into reservoir states
    """

    # Constructor
    def __init__(self, input_dim, noise_generator, *args, **kwargs):
        """
        Constructor
        :param input_dim: Node's input dimension.
        :param output_dim: Node's output dimension.
        """
        super(NoiseInjection, self).__init__(
            input_dim=input_dim,
            output_dim=input_dim,
            *args,
            **kwargs
        )

        # Noise generator
        self._noise_generator = noise_generator
    # end __init__

    # Forward
    def forward(self, x):
        """
        Forward
        :return:
        """
        return x + self._noise_generator(self._input_dim)
    # end forward

# end NoiseInjection
