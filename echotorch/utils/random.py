# -*- coding: utf-8 -*-
#
# File : echotorch/utils/random.py
# Description : Utility function for random number generators
# Date : 30th of April, 2021
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
import random
import numpy as np
import torch


# Set random seed
def manual_seed(seed):
    """
    Set manual seed for pytorch and numpy
    :param seed: Seed for pytorch and numpy
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
# end manual_seed
