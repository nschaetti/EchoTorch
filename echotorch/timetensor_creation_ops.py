# -*- coding: utf-8 -*-
#
# File : echotorch/timetensor_creation_ops.py
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
from typing import Optional, Tuple, Union, Any
import numpy as np
import torch

# Import local
from .timetensor import TimeTensor


# Constructs a timetensor with data.
def timetensor(
        data: Any,
        time_dim: Optional[int] = 0,
        time_lengths: Optional[torch.LongTensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False,
        pin_memory: Optional[bool] = False
) -> 'TimeTensor':
    """
    Constructs a tensor with data.
    @param data: Initial data for the timetensor. Can be a list, tuple, numpy ndarray, scalar, and other types.
    @param time_dim: Index of the time dimension
    @param time_lengths:
    @param dtype: The desired data type of returned timetensor. Default: if None, infers data type from data.
    @param device: ...
    @param requires_grad: Requires gradient computation?
    @param pin_memory:
    @return: A TimeTensor object
    """
    # Data
    if isinstance(data, torch.Tensor):
        src_data = data.clone().detach()
    elif isinstance(data, TimeTensor):
        src_data = data.tensor.clone().detach()
    else:
        src_data = torch.tensor(
            data,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory
        )
    # end if

    return TimeTensor.new_timetensor(
        src_data,
        time_dim=time_dim,
        time_lengths=time_lengths
    )
# end timetensor


# Convert data into an echotorch.TimeTensor.
def as_timetensor(
        data: Any,
        time_lengths: Optional[torch.LongTensor] = None,
        time_dim: Optional[int] = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
) -> 'TimeTensor':
    """
    Convert data into an echotorch.TimeTensor
    @param time_dim:
    @param time_lengths:
    @param data:
    @param dtype:
    @param device:
    @return:
    """
    return TimeTensor.new_timetensor(
        torch.as_tensor(
            data,
            dtype=dtype,
            device=device
        ),
        time_dim=time_dim,
        time_lengths=time_lengths
    )
# end as_timetensor


# From Numpy
def from_numpy(
        ndarray: np.ndarray,
        time_lengths: Optional[torch.LongTensor] = None,
        time_dim: Optional[int] = 0,
) -> TimeTensor:
    """
    Creates a TimeTensor from a numpy.ndarray.

    @param time_dim:
    @param time_lengths:
    @param ndarray:
    @return:
    """
    return TimeTensor.new_timetensor(
        torch.from_numpy(ndarray),
        time_dim=time_dim,
        time_lengths=time_lengths
    )
# end from_numpy


# Returns filled time tensor
def full(
        size: Tuple[int],
        fill_value: Union[int, float],
        time_length: Union[int, torch.LongTensor],
        out: Optional[TimeTensor] = None,
        dtype: Optional[torch.dtype] = None,
        layout: Optional[torch.layout] = torch.strided,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False
) -> 'TimeTensor':
    """
    Returns a TimeTensor of size size and time length time_length filled with fill_value. By default,
    the returned Tensor has the same torch.dtype and torch.device as this tensor.
    @param out:
    @param layout:
    @param size: Size of the time series (without time dimension)
    @param time_length: Size of the time dimension
    @param fill_value: Value used to fill the timeseries
    @param dtype:
    @param device:
    @param requires_grad:
    @return: A TimeTensor
    """
    # Size
    if out is not None:
        out = TimeTensor.new_timetensor_with_func(
            size=size,
            func=torch.full,
            time_length=time_length,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            fill_value=fill_value,
            layout=layout
        )
        return out
    else:
        return TimeTensor.new_timetensor_with_func(
            size=size,
            func=torch.full,
            time_length=time_length,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            fill_value=fill_value,
            layout=layout
        )
    # end if
# end full


# Returns empty tensor
def empty(
        size: Tuple[int],
        time_length: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False
) -> 'TimeTensor':
    """
    Returns a TimeTensor of size size and time length time_length filled with uninitialized data. By default,
    the returned TimeTensor has the same torch.dtype and torch.device as this tensor.
    @param size:
    @param time_length:
    @param dtype:
    @param device:
    @param requires_grad:
    """
    return TimeTensor.new_timetensor_with_func(
        size,
        func=torch.empty,
        time_length=time_length,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
# end empty


# Returns time tensor filled with ones
def ones(
        size: Tuple[int],
        time_length: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False
) -> 'TimeTensor':
    """
    Returns a TimeTensor of size size filled with 1. By default, the returned TimeTensor has the same
    torch.dtype and torch.device as this tensor.
    @param size:
    @param time_length:
    @param dtype:
    @param device:
    @param requires_grad:
    """
    return TimeTensor.new_timetensor_with_func(
        size,
        func=torch.ones,
        time_length=time_length,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
# end ones


# Returns time tensor filled with zeros
def zeros(
        size: Tuple[int],
        time_length: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False
) -> 'TimeTensor':
    """
    Returns a TimeTensor of size size filled with 0.
    @param size:
    @param time_length:
    @param dtype:
    @param device:
    @param requires_grad:
    """
    return TimeTensor.new_timetensor_with_func(
        size=size,
        func=torch.zeros,
        time_length=time_length,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
# end new_zeros
