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

# Local imports
from .timetensor import TimeTensor


# Concatenate on time dim
def tcat(
        *tensors: Tuple[TimeTensor]
) -> TimeTensor:
    r"""Concatenate a given list of ``n`` tensors on the time dimension. All timetensors
    must have the same shape (except the time dimensions) and the same time dimension
    specified or be emoty.

    ``echotorch.tcat()`` is the inverse of ``echotorch.tsplit()`` and ``echotorch.tchunk``.

    Parameters:
        tensors (sequence of TimeTensors) - any python sequence of timetensors of the same type. If timetensors are not empty, they must have the same shape, except the time dimension, and the same time dimension specified.

    Key Arguments:
        out (TimeTensor, optional) - the output timetensor.

    Example::

        >>> x = echotorch.randn(2, time_length=20)
        >>> x
        timetensor([[....]])
        >>> echotorch.tcat((x, x, x))
        timetensor([[...]])

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
    r"""Returns True if `obj` is an EchoTorch timetensor.

    Note that this function is simply doing ``isinstance(obj, TimeTensor)``.
    Using that ``isinstance`` check is better for typechecking with mypy,
    and more explicit - so it's recommended to use that instead of
    ``is_timetensor``.

    Args:
        obj (Object): Object to test
    Example::

        >>> x = echotorch.timetensor([1,2,3], time_dim=0)
        >>> echotorch.is_timetensor(x)
        True

    """
    return isinstance(obj, TimeTensor)
# end is_timetensor

