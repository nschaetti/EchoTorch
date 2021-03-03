# -*- coding: utf-8 -*-
#
# File : echotorch/utility_functions.py
# Description : Utility functions for EchoTorch
# Date : 25th of January, 2021
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
from .tensor import TimeTensor


# Create a time tensor
def timetensor(data, time_dim=0):
    """
    Create a temporal tensor
    """
    return TimeTensor(data, time_dim=time_dim)
# end timetensor


# Concatenate on time dim
def timecat(tensor1, tensor2):
    """
    Concatenate
    """
    if tensor1.time_dim == tensor2.time_dim and tensor1.ndim == tensor2.ndim:
        return torch.cat((tensor1, tensor2), dim=tensor1.time_dim)
    else:
        raise Exception(
            "Tensor 1 and 2 must have the same number of dimension and the same time dimension (here {}/{} "
            "and {}/{}".format(tensor1.ndim, tensor1.time_dim, tensor2.ndim, tensor2.time_dim)
        )
    # end if
# end timecat
