# -*- coding: utf-8 -*-
#
# File : echotorch/tensor_utils.py
# Description : Tensor utility functions
# Date : 23th of February, 2021
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
import torch
import numpy as np
from .tensor import TimeTensor


# From numpy
def from_numpy(np_array: np.array, time_dim=0, with_batch=False) -> TimeTensor:
    """
    Create a TimeTensor from a numpy array
    """
    # Check dims
    if with_batch and time_dim == 0:
        raise Exception("Time dim and batch dim must be different (here {} and {})".format(time_dim, 0))
    # end if

    return TimeTensor(np_array, time_dim=time_dim, with_batch=with_batch)
# end from_numpy

