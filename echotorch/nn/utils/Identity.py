# -*- coding: utf-8 -*-
#
# File : echotorch/nn/Identity.py
# Description : An Leaky-Integrated Echo State Network layer.
# Date : 09th of April, 2018
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


# Identity layer
class Identity(nn.Module):
    """
    Identity layer
    """

    # Forward
    def forward(self, x):
        """
        Forward
        :return:
        """
        return x
    # end forward

# end Identity
