# -*- coding: utf-8 -*-
#
# File : echotorch/timetensor_creation.py
# Description : TimeTensor creation helper functions
# Date : 27th of Jully, 2021
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
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>
# University of Geneva <nils.schaetti@unige.ch>


# Imports
from typing import Optional, Tuple, Union
import torch

# Import local
from .timetensor import TimeTensor


# Returns a new TimeTensor with data as the tensor data.
def timetensor(
        data: torch.Tensor,
        time_dim: Optional[int] = 0,
        time_first: Optional[bool] = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False
) -> 'TimeTensor':
    """
    Returns a new TimeTensor with data as the tensor data. If data is already a timetensor, a copy is made,
    otherwise data is converted to a tensor and copied with the as_tensor function.
    @param data: Data as a torch tensor
    @param time_dim: Position of the time dimension
    @param time_first:
    @param dtype: Torch data type
    @param device: Destination device
    @param requires_grad: Requires gradient computation?
    @return: A TimeTensor object
    """
    return TimeTensor.new_timetensor(
        data,
        time_dim=time_dim,
        time_first=time_first,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad
    )
# end timetensor


# Returns filled time tensor
def full(
        size: Tuple[int],
        time_length: int,
        fill_value: Union[int, float],
        time_first: Optional[bool] = True,
        dtype: torch.dtype = None,
        device: torch.device = None,
        requires_grad: bool = False
) -> 'TimeTensor':
    """
    Returns a TimeTensor of size size and time length time_length filled with fill_value. By default,
    the returned Tensor has the same torch.dtype and torch.device as this tensor.
    @param size: Size of the time series (without time dimension)
    @param time_length: Size of the time dimension
    @param fill_value: Value used to fill the timeseries
    @param time_first:
    @param dtype:
    @param device:
    @param requires_grad:
    @return: A TimeTensor
    """
    # Size
    return TimeTensor.new_full(
        size,
        time_length=time_length,
        time_first=time_first,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        fill_value=fill_value
    )
# end full


# Returns empty tensor
def empty(
        size: Tuple[int],
        time_length: int,
        time_first: Optional[bool] = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False
) -> 'TimeTensor':
    """
    Returns a TimeTensor of size size and time length time_length filled with uninitialized data. By default,
    the returned TimeTensor has the same torch.dtype and torch.device as this tensor.
    @param size:
    @param time_length:
    @param time_first:
    @param dtype:
    @param device:
    @param requires_grad:
    """
    return TimeTensor.new_empty(
        size,
        time_length=time_length,
        time_first=time_first,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
# end empty


# Returns time tensor filled with ones
def ones(
        size: Tuple[int],
        time_length: int,
        time_first: Optional[bool] = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False
) -> 'TimeTensor':
    """
    Returns a TimeTensor of size size filled with 1. By default, the returned TimeTensor has the same
    torch.dtype and torch.device as this tensor.
    @param size:
    @param time_length:
    @param time_first:
    @param dtype:
    @param device:
    @param requires_grad:
    """
    return TimeTensor.new_ones(
        size,
        time_length=time_length,
        time_first=time_first,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
# end ones


# Returns time tensor filled with zeros
def zeros(
        size: Tuple[int],
        time_length: int,
        time_first: Optional[bool] = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False
) -> 'TimeTensor':
    """
    Returns a TimeTensor of size size filled with 0.
    @param size:
    @param time_length:
    @param time_first:
    @param dtype:
    @param device:
    @param requires_grad:
    """
    return TimeTensor.new_zeros(
        size,
        time_length=time_length,
        time_first=time_first,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
# end new_zeros
