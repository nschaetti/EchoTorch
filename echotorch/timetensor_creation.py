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

from .timetensor import TimeTensor


# Returns a new TimeTensor with data as the tensor data.
def timetensor(
        data: torch.Tensor,
        time_dim: Optional[int] = 0,
        with_batch: Optional[bool] = False,
        with_channels: Optional[bool] = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False
) -> 'TimeTensor':
    """
    Returns a new TimeTensor with data as the tensor data. If data is already a timetensor, a copy is made,
    otherwise data is converted to a tensor and copied with the as_tensor function.
    @param data: Data as a torch tensor
    @param time_dim: Position of the time dimension
    @param with_batch: Batch dimension included
    @param with_channels:
    @param dtype: Torch data type
    @param device: Destination device
    @param requires_grad: Requires gradient computation?
    @return: A TimeTensor object
    """
    return TimeTensor.new_timetensor(
        data,
        time_dim=time_dim,
        with_batch=with_batch,
        with_channels=with_channels,
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
        n_channels: Optional[int] = None,
        batch_size: Optional[int] = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
        requires_grad: bool = False
) -> 'TimeTensor':
    """
    Returns a TimeTensor of size size and time length time_length filled with fill_value. By default,
    the returned Tensor has the same torch.dtype and torch.device as this tensor.
    @param size:
    @param time_length:
    @param n_channels:
    @param batch_size:
    @param fill_value:
    @param with_batch:
    @param dtype:
    @param device:
    @param requires_grad:
    @return: A TimeTensor
    """
    # Size
    return TimeTensor.new_full(
        size=size,
        time_length=time_length,
        n_channels=n_channels,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        fill_value=fill_value
    )
    return
# end full


# Returns empty tensor
def new_empty(
        size, time_length, n_channels, batch_size, with_batch=False, dtype=None, device=None,
        requires_grad=False
) -> 'TimeTensor':
    """Returns a TimeTensor of size size and time length time_length filled with uninitialized data. By default,
    the returned TimeTensor has the same torch.dtype and torch.device as this tensor."""
    return TimeTensor.new_timetensor_with_func(
        size=size,
        func=torch.Tensor.new_empty,
        time_length=time_length,
        n_channels=n_channels,
        batch_size=batch_size,
        with_batch=with_batch,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )

# end new_empty

# Returns time tensor filled with ones
def new_ones(
        size, time_length, n_channels, batch_size, with_batch=False, dtype=None, device=None,
        requires_grad=False
) -> 'TimeTensor':
    """Returns a TimeTensor of size size filled with 1. By default, the returned TimeTensor has the same
    torch.dtype and torch.device as this tensor."""
    return TimeTensor.new_timetensor_with_func(
        size=size,
        func=torch.Tensor.new_ones,
        time_length=time_length,
        n_channels=n_channels,
        batch_size=batch_size,
        with_batch=with_batch,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )

# end new_ones

# Returns time tensor filled with zeros
def new_zeros(
        size, time_length, n_channels, batch_size, with_batch=False, dtype=None, device=None,
        requires_grad=False
):
    """Returns a TimeTensor of size size filled with 0.
    """
    return TimeTensor.new_timetensor_with_func(
        size=size,
        func=torch.Tensor.new_zeros,
        time_length=time_length,
        n_channels=n_channels,
        batch_size=batch_size,
        with_batch=with_batch,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )

# end new_zeros

# Returns new time tensor with a specific function
def new_timetensor_with_func(
        size, func, time_length, n_channels, batch_size, with_batch=False, dtype=None, device=None,
        requires_grad=False, **kwargs
) -> 'TimeTensor':
    """Returns a new time tensor with a specific function to generate the data.
    """
    # Size
    tt_size = [batch_size, n_channels, time_length] if with_batch else [n_channels, time_length]
    tt_size += size

    # Create TimeTensor
    return TimeTensor(
        func(size=tuple(tt_size), **kwargs),
        time_dim=1 if with_batch else 0,
        with_batch=with_batch,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad
    )
# end new_timetensor_with_func

