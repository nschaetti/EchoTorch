# -*- coding: utf-8 -*-
#
# File : echotorch/timetensor_utility.py
# Description : Utility functions for TimeTensors
# Date : 1st of August, 2021
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

from .base_ops import tcat, cat, tindex_select, is_timetensor
from .timetensors import TimeTensor
import torch


def eqtd(inputs):
    if len(inputs) < 2:
        return True
    time_dim = inputs[0].time_dim
    ndim = inputs[0].ndim
    return all(tt.time_dim == time_dim and tt.ndim == ndim for tt in inputs)


def eq_time(inputs):
    if len(inputs) < 2:
        return True
    time_dim = inputs[0].time_dim
    time_length = inputs[0].tlen
    return all(tt.time_dim == time_dim and tt.tlen == time_length for tt in inputs)


def eq_chan(inputs):
    return eqtd(inputs)


def similar_timetensors(inputs):
    return eqtd(inputs) and eq_time(inputs) and eq_chan(inputs)


# Unsqueeze
def unsqueeze(
        input: TimeTensor,
        dim: int
) -> TimeTensor:
    """
    Returns a new timetensor with a dimension of size one inserted at the specified position.

    The returned timetensor shares the same underlying data with this timetensor.

    A dim value within the range [-input.dim() - 1, input.dim() + 1) can be used.
    Negative dim will correspond to unsqueeze() applied at dim = dim + input.dim() + 1.

    @param input: the input tensor
    @param dim: the index at which to insert the singleton dimension
    @return: the new timetensor
    """
    return torch.unsqueeze(input, dim)
# end unsqueeze


# Transform a timetensor to a tensor
def to_tensor(
        input: TimeTensor
) -> torch.Tensor:
    """
    Transform a timetensor to a tensor
    @param input: The input timetensor
    @return: The tensor
    """
    return input.tensor
# end to_tensor
