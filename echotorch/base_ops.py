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
from echotorch import TimeTensor


# region CREATION_OPS

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

# endregion CREATION_OPS

# region DISTRIBUTION_OPS

# Returns a timetensor filled with random numbers from a uniform distribution on the interval [0, 1)[0,1)
def rand(
        size: Tuple[int],
        time_length: int,
        out: Optional[TimeTensor] = None,
        dtype: Optional[torch.dtype] = None,
        layout: Optional[torch.layout] = torch.strided,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False
) -> TimeTensor:
    r"""Returns a timetensor filled with random numbers from a uniform distribution on the interval [0, 1)[0,1)

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
        >>> x = echotorch.rand((1), time_length=10)
        timetensor([[1.], [1.], [1.], [1.], [1.]])
    """
    return TimeTensor.new_timetensor_with_func(
        size,
        func=torch.rand,
        time_length=time_length,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        layout=layout
    )
# end rand

# endregion DISTRIB_OPS

# region UTILITY_OPS

# Concatenate on time dim
def tcat(
        *tensors: Tuple[TimeTensor]
) -> TimeTensor:
    r"""Concatenate a given list of ``n`` timetensors or tensor on the time dimension. All timetensors
    must have the same shape (except the time dimensions) and the same time dimension
    specified or be empty. If PyTorch tensors are in the sequence, they must have the same shape and they
    will be concatenated on the same dimension as specified as time dimension in  the timetensors. The concatenation
    will fail if there is only PyTorch tensors in the sequence.

    ``echotorch.tcat()`` is the inverse of ``echotorch.tsplit()`` and ``echotorch.tchunk()``.

    :param tensors: A sequence of timetensors or tensors of the same type, same time dimension and same shape.
    :return: The timetensors/tensors concatenated in a single timetensor.
    :return: ``TimeTensor``

    Parameters:
        **tensors** (sequence of ``TimeTensors``) - any python sequence of timetensors or PyTorch tensors of the same
        type. If timetensors are not empty, they must have the same shape, except the time dimension, and the same
        time dimension specified. If PyTorch tensors are in the sequence, they must have the same shape and they
        will be concatenated on the same dimension as specified as time dimension in  the timetensors. The concatenation
        will fail if there is only PyTorch tensors in the sequence.

    Key Arguments:
        **out** (``TimeTensor``, optional) - the output timetensor.

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
def is_timetensor(obj) -> bool:
    r"""Returns True if `obj` is an EchoTorch timetensor.

    Note that this function is simply doing ``isinstance(obj, TimeTensor)``.
    Using that ``isinstance`` check is better for typechecking with mypy,
    and more explicit - so it's recommended to use that instead of
    ``is_timetensor``.

    :param obj: The object to test
    :type obj: `object`
    :return: True if `obj` is an EchoTorch timetensor
    :rtype: bool

    Example::

        >>> x = echotorch.timetensor([1,2,3], time_dim=0)
        >>> echotorch.is_timetensor(x)
        True

    """
    return isinstance(obj, TimeTensor)
# end is_timetensor

# endregion UTILITY_OPS
