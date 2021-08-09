# -*- coding: utf-8 -*-
#
# File : echotorch/timetensors/timetensor_creation_ops.py
# Description : TimeTensor creation helper functions
# Date : 27th of July, 2021
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
    r"""Construct a timetensor with data as tensor, timetensor, list, etc.

    :param data: Data as tensor, timetensor, list or Numpay array.
    :type data: Tensor, TimeTensor, List, Numpy Array
    :param time_dim:
    :type time_dim: Integer
    :param time_lengths: Length of the timeseries, or lengths as a LongTensor if the timetensor contained multiple timeseries with different lengths.
    :type time_lengths: Integer or LongTensor
    :param dtype: Torch Tensor data type
    :type dtype: torch.dtype
    :param device: Destination device (cpu, gpu, etc)
    :type device: torch.device
    :param requires_grad: Compute gradients?
    :type requires_grad: Boolean
    :param pin_memory:
    :type pin_memory: Boolean
    :return: The ``TimeTensor`` created from ``data``
    :rtype: TimeTensor

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


# Convert data into a TimeTensor
def as_timetensor(
        data: Any,
        time_lengths: Optional[torch.LongTensor] = None,
        time_dim: Optional[int] = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
) -> 'TimeTensor':
    r"""Convert data into a ``TimeTensor``.

    :param data: Data to convert as TimeTensor, Tensor, List or Numpy array.
    :type data: TimeTensor, Tensor, List, Numpy array
    :param time_lengths: Length of the timeseries, or lengths as a LongTensor if the timetensor contained multiple timeseries with different lengths.
    :type time_lengths: Integer or LongTensor
    :param time_dim: The index of the time dimension.
    :type time_dim: Integer
    :param dtype: Tensor data type
    :type dtype: torch.dtype
    :param device: Destination device
    :type device: torch.device
    :return: The ``TimeTensor`` created from data.
    :rtype: TimeTensor

    Example::
        >>> x = echotorch.as_timetensor([[0], [1], [2]], time_dim=0)
        >>> x.tsize()
        torch.Size([1])
        >>> x.tlen
        3

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
    r"""Creates a TimeTensor from a ``numpy.ndarray``.

    :param time_dim: Index of the time dimension.
    :type time_dim: Integer
    :param time_lengths:
    :type time_lengths: ``int`` or ``LongTensor``
    :param ndarray: The numpy array
    :type ndarray: ``numpy.array`` or ``numpay.ndarray``
    :return: ``TimeTensor`` created from Numpy data.
    :rtype: TimeTensor

    Examples::
        >>> x = echotorch.from_numpy(np.zeros((100, 2)), time_dim=0)
        >>> x.size()
        torch.Size([100, 2])
        >>> x.tlen
        100

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
    r"""Returns a TimeTensor of size size and time length time_length filled with fill_value.

    :param size: Size
    :type size: Tuple[int]
    :param fill_value: the value to fill the output timetensor with.
    :type fill_value: Scalar
    :param time_length: Length of the timeseries
    :type time_length: int
    :param dtype: ``TimeTensor`` data type
    :type dtype: torch.dtype
    :param device: Destination device
    :type device: torch.device
    :param requires_grad: Activate gradient computation
    :type requires_grad: bool
    :return: A ``TimeTensor`` of size size filled with zeros
    :rtype: ``TimeTensor``

    Example::
        >>> x = echotorch.full((2, 2), time_length=100)
        >>> x.size()
        torch.Size([100, 2, 2])
        >>> x.tsize()
        torch.Size([2, 2])
        >>> x.tlen
        100
        >>> echotorch.full((), time_length=5)
        timetensor([ 1., 1., 1., 1., 1.])
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
    r"""Returns a :class:`TimeTensor` of size ``size`` and time length ``time_length`` filled with uninitialized data.

    :param size: Size
    :type size: Tuple[int]
    :param time_length: Length of the timeseries
    :type time_length: int
    :param dtype: ``TimeTensor`` data type
    :type dtype: torch.dtype
    :param device: Destination device
    :type device: torch.device
    :param requires_grad: Activate gradient computation
    :type requires_grad: bool
    :return: A ``TimeTensor`` of size size filled with zeros
    :rtype: ``TimeTensor``

    Example::
        >>> x = echotorch.empty((2, 3), time_length=1, dtype=torch.int32, device = 'cuda')
        >>> echotorch.empty_like(x)
        timetensor([[[1., 1., 1.],
                     [1., 1., 1.]]], device='cuda:0', dtype=torch.int32)
    """
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
    r"""Returns a ``TimeTensor`` of size ``size`` filled with 1.

    :param size: Size
    :type size: Tuple[int]
    :param time_length: Length of the timeseries
    :type time_length: int
    :param dtype: ``TimeTensor`` data type
    :type dtype: torch.dtype
    :param device: Destination device
    :type device: torch.device
    :param requires_grad: Activate gradient computation
    :type requires_grad: bool
    :return: A ``TimeTensor`` of size size filled with zeros
    :rtype: ``TimeTensor``

    Example::
        >>> x = echotorch.ones((2, 2), time_length=100)
        >>> x.size()
        torch.Size([100, 2, 2])
        >>> x.tsize()
        torch.Size([2, 2])
        >>> x.tlen
        100
        >>> echotorch.ones((), time_length=5)
        timetensor([ 1., 1., 1., 1., 1.])
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
    r"""Returns a ``TimeTensor`` of size ``size`` filled with 0.

    :param size: Size
    :type size: Tuple[int]
    :param time_length: Length of the timeseries
    :type time_length: int
    :param dtype: ``TimeTensor`` data type
    :type dtype: torch.dtype
    :param device: Destination device
    :type device: torch.device
    :param requires_grad: Activate gradient computation
    :type requires_grad: bool
    :return: A ``TimeTensor`` of size size filled with zeros
    :rtype: ``TimeTensor``

    Example::
        >>> x = echotorch.zeros((2, 2), time_length=100)
        >>> x.size()
        torch.Size([100, 2, 2])
        >>> x.tsize()
        torch.Size([2, 2])
        >>> x.tlen
        100
        >>> echotorch.zeros((), time_length=5)
        timetensor([ 0., 0., 0., 0., 0.])
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
