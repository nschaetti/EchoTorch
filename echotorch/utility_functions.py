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
from typing import Tuple, Union, Any
import torch
# from .tensor import TimeTensor
from .timetensor import TimeTensor


# Create a time tensor
def timetensor(
        data, dtype=None, device=None, required_grad=False, pin_memory=False, time_dim=0, with_batch=False
) -> TimeTensor:
    """
    Create a temporal tensor
    """
    return TimeTensor(
        data, time_dim=time_dim, with_batch=with_batch, dtype=dtype, device=device, required_grad=required_grad,
        pin_memory=pin_memory
    )
# end timetensor


# Concatenate on time dim
def tcat(tensors: Tuple[TimeTensor]) -> Union[TimeTensor, Any]:
    """
    Concatenate
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


# Concatenate time-related dimension
def cat(tensors: Tuple[TimeTensor], dim: int = 0) -> Union[TimeTensor, Any]:
    """Concatenate time-related dimensions
    """
    # None
    if len(tensors) == 0:
        return None
    # end if

    # First time dim and ndim
    time_dim = tensors[0].time_dim
    ndim = tensors[0].ndim
    tlen = tensors[0].tlen

    # Check all tensor
    for tensor in tensors:
        if tensor.time_dim != time_dim or tensor.ndim != ndim or tensor.tlen != tlen:
            raise Exception(
                "Tensor 1 and 2 must have the same number of dimension, the same time dimension and the same "
                "time length (here {}/{}, {}/{} and {}/{})".format(
                    ndim,
                    tensor.ndim,
                    time_dim,
                    tensor.time_dim,
                    tlen,
                    tensor.tlen
                )
            )
        # end if
    # end if

    # Time tensor
    return torch.cat(tensors, dim=time_dim+1+dim)
# end cat


# Select time index in tensor
def tindex_select(input: TimeTensor, indices: Union[torch.IntTensor, torch.LongTensor]) -> TimeTensor:
    """Select time index in time tensor
    """
    return torch.index_select(
        input,
        input.time_dim,
        indices
    )
# end tindex_select


# Tensor filled with zeros
def zeros(
        size, time_length, n_channels, batch_size, with_batch=False, out=None, dtype=None, layout=torch.strided,
        device=None, requires_grad=False
) -> TimeTensor:
    """Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size.
    """
    if out is None:
        return TimeTensor.new_zeros(
            size=size,
            time_length=time_length,
            n_channels=n_channels,
            batch_size=batch_size,
            with_batch=with_batch,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad
        )
    else:
        out = TimeTensor.new_zeros(
            size=size,
            time_length=time_length,
            n_channels=n_channels,
            batch_size=batch_size,
            with_batch=with_batch,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad
        )
    # end if
# end zeros
