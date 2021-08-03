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

# Imports
from typing import Tuple, Union, Any, List
import torch

# Local imports
from .timetensor import TimeTensor


# Concatenate on time dim
def tcat(
        *tensors: Tuple[TimeTensor]
) -> Union[TimeTensor, Any]:
    """
    Concatenate on time dimension
    :@param tensors: Tensor to concatenate
    :@return: Concatenated tensors
    """
    # None
    if len(tensors) == 0:
        return None
    # end if

    # First time dim and ndim
    time_dim = tensors[0].time_dim
    ndim = tensors[0].ndim

    # Check all tensor
    for tensor in tensors:
        if tensor.time_dim != time_dim or tensor.ndim != ndim:
            raise Exception(
                "Tensor 1 and 2 must have the same number of dimension and the same time dimension (here {}/{} "
                "and {}/{}".format(ndim, time_dim, tensor.ndim, tensor.time_dim)
            )
        # end if
    # end if

    # Time tensor
    return torch.cat(tensors, dim=time_dim)
# end tcat


# Is the index of the time dimension the same for all time-tensors
def eqtd(
        inputs: Union[List[TimeTensor], Tuple[TimeTensor]]
) -> bool:
    """
    Is the index of the time dimension the same for all time-tensors
    @param inputs: A list or tuple of time-tensors
    @return: True if all indices of time dimensions are the same
    """
    # Check number of items
    if len(inputs) < 2:
        return True
    # end if

    # First time dim and ndim
    time_dim = inputs[0].time_dim
    ndim = inputs[0].ndim

    # Check all tensor
    for ttensor in inputs:
        if ttensor.time_dim != time_dim or ttensor.ndim != ndim:
            return False
        # end if
    # end if

    return True
# end eqtd


# Have all time-tensors contained in a list have the same time length
def eq_time(
        inputs: Union[List[TimeTensor], Tuple[TimeTensor]]
) -> bool:
    """
    Have all time-tensors contained in a list have the same time length
    @param inputs:
    @return:
    """
    # Check number of items
    if len(inputs) < 2:
        return True
    # end if

    # First time dim and ndim
    time_dim = inputs[0].time_dim
    time_length = inputs[0].tlen

    # Check all tensor
    for ttensor in inputs:
        if ttensor.time_dim != time_dim or ttensor.tlen != time_length:
            return False
        # end if
    # end if

    return True
# end eq_time


# Check if all time-tensors in the list/tuple have the same channel dimension position
def eq_chan(
        inputs: Union[List[TimeTensor], Tuple[TimeTensor]]
) -> bool:
    """
    Check if all time-tensors in the list/tuple have the same channel dimension position
    @param inputs:
    @return:
    """
    # Check number of items
    if len(inputs) < 2:
        return True
    # end if

    # Channel side of the first element
    time_first = inputs[0].time_first

    # Check all tensor
    for ttensor in inputs:
        if ttensor.time_first != time_first:
            return False
        # end if
    # end if

    return True
# end eq_chan


# Check if time-tensors in a list/tuple are similar (same time dimension, length, and channel position)
def similar_timetensors(
        inputs: Union[List[TimeTensor], Tuple[TimeTensor]]
) -> bool:
    """
    Check if time-tensors in a list/tuple are similar (same time dimension, length, and channel position)
    @param inputs:
    @return:
    """
    return eqtd(inputs) and eq_time(inputs) and eq_chan(inputs)
# end similar_timetensors


# Concatenate channel dimension
def cat(
        inputs: Union[Tuple[TimeTensor], List[TimeTensor]],
        dim: int = 0
) -> Union[TimeTensor, Any]:
    """
    Concatenate time-related dimensions
    """
    # None
    if len(inputs) == 0:
        return None
    elif len(inputs) == 1:
        return inputs
    # end if

    # First time dim and ndim
    time_dim = inputs[0].time_dim

    # Check that all tensor have the same time dimension index
    # and the same time length
    if not eqtd(inputs) or not eq_time(inputs):
        raise Exception(
            "Tensor 1 and 2 must have the same number of dimension, the same time dimension and the same "
            "time length (here {}/{}, {}/{} and {}/{})".format(
                inputs[0].ndim,
                inputs[1].ndim,
                inputs[0].ndim,
                inputs[1].time_dim,
                inputs[0].ndim,
                inputs[1].tlen
            )
        )
    # end if

    # Time tensor
    return torch.cat(inputs, dim=time_dim+1+dim)
# end cat


# Select time index in tensor
def tindex_select(
        input: TimeTensor,
        indices: Union[torch.IntTensor, torch.LongTensor]
) -> TimeTensor:
    """
    Select time index in time tensor
    """
    return torch.index_select(
        input,
        input.time_dim,
        indices
    )
# end tindex_select


# Is timetensor
def is_timetensor(self, obj) -> bool:
    """
    Returns True if obj is a PyTorch tensor.
    @param obj: Object to check
    @return:
    """
    return isinstance(obj, TimeTensor)
# end is_timetensor


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

