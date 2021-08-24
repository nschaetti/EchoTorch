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
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False,
        pin_memory: Optional[bool] = False
) -> 'TimeTensor':
    r"""Construct a :class:`TimeTensor` with given data as tensor, timetensor, list, etc.

    .. warning::
        Similarly to ``torch.tensor()``, :func:`echotorch.timetensor()` copies the data. For more information on how
        to avoid copy, check the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.tensor.html#torch.tensor>`__.

    .. warning::
        Like ``torch.tensor()``, :func:`echotorch.timetensor()` reads out data and construct a leaf variable. Check
        the `PyTorch documentation on torch.tensor() <https://pytorch.org/docs/stable/generated/torch.tensor.html#torch.tensor>`__ for more information.

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.tensor.html#torch.tensor>`__ on ``tensor()`` for more informations.

    :param data: data for the wrapped :class:`torch.Tensor` as a tensor, timetensor, list or Numpy array.
    :type data: array_like
    :param time_dim: the index of the time dimension (default: 0).
    :type time_dim: ``int``, optional
    :param dtype: the desired data type of the wrapped tensor (default: None, infered from ``data``).
    :type dtype: :class:`torch.dtype`, optional
    :param device: the estination device of the wrapped tensor (default: None, current device, see ``torch.set_default_tensor_type()``).
    :type device: :class:`torch.device`, optional
    :param requires_grad: Should operations been recorded by autograd for this timetensor?
    :type requires_grad: `bool`, optional
    :param pin_memory: If set, returned timetensor would be allocated in the pinned memory. Works only for CPU timetensors (default: ``False``)
    :type pin_memory: `bool`, optional

    Example:

        >>> echotorch.timetensor([1, 2, 3, 4], device='cuda:0')
        timetensor([1, 2, 3, 4], device='cuda:0', time_dim: 0)
    """
    # Data
    if isinstance(data, torch.Tensor):
        src_data = data.clone().detach().requires_grad_(requires_grad)
    elif isinstance(data, TimeTensor):
        src_data = data.tensor.clone().detach().requires_grad_(requires_grad)
    else:
        src_data = torch.tensor(
            data,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory
        )
    # end if

    # Set parameters
    src_data = src_data.to(dtype=dtype, device=device)
    if pin_memory: src_data = src_data.pin_memory()

    # Create timetensor
    return TimeTensor.new_timetensor(
        src_data,
        time_dim=time_dim
    )
# end timetensor


# Convert data into a TimeTensor
def as_timetensor(
        data: Any,
        time_dim: Optional[int] = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
) -> 'TimeTensor':
    r"""Convert data into a :class:`TimeTensor`. If a :class:`torch.Tensor` or a :class:`TimeTensor` is given as data,
    no copy will be made, otherwise a new :class:`torch.Tensor` will be wrapped with computational graph retained if
    the tensor has ``requires_grad`` set to ``True``. If the data comes frome Numpy (:class:`ndarray`) with the same
    *dtype* and is on the cpu, the no copy will be made. This behavior is similar to :func:`torch.as_tensor()`. See
    the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.as_tensor.html#torch.as_tensor>`__ for more information.

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.as_tensor.html#torch.as_tensor>`__ on ``as_tensor()`` for more informations.

    :param data: data to convert for the wrapper tensor as :class:`TimeTensor`, Tensor, List, scalar or Numpy array.
    :type data: array-like
    :param time_dim: the index of the time dimension.
    :type time_dim: ``int``, optional
    :param dtype: the desired data type of the wrapped tensor (default: None, infered from ``data``).
    :type dtype: :class:`torch.dtype`, optional
    :param device: the estination device of the wrapped tensor (default: None, current device, see ``torch.set_default_tensor_type()``).
    :type device: :class:`torch.device`, optional

    Example:

        >>> x = echotorch.as_timetensor([[0], [1], [2]], time_dim=0)
        >>> x
        timetensor([[0],
                    [1],
                    [2]], time_dim: 0)
        >>> x.csize()
        torch.Size([1])
        >>> x.bsize()
        torch.Size([])
        >>> x.tlen
        3
    """
    return TimeTensor.new_timetensor(
        torch.as_tensor(
            data,
            dtype=dtype,
            device=device
        ),
        time_dim=time_dim
    )
# end as_timetensor


# Sparse COO timetensor
def sparse_coo_timetensor(
        indices,
        values,
        time_dim: Optional[int] = 0,
        size=None,
        dtype=None,
        device=None,
        requires_grad=False
) -> TimeTensor:
    r"""Construct a :class:`TimeTensor` with a wrapped `sparse *tensor* in COO(rdinate) format <https://pytorch.org/docs/stable/sparse.html#sparse-coo-docs>`__ with specified values at the given indices.

    .. note::
        The contained tensor is an uncoalesced tensor.

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html#torch.sparse_coo_tensor>`__ on ``sparse_coo_tensor()`` for more informations.

    :param indices: the data indices for the wrapped tensor as a list, tuple, Numpy ``ndarray``, scalar, etc. Indices will casted to ``torch.LongTensor``. Indices are the coordinates of data inside the matrix.
    :type indices: array_like
    :param values: the data values for the wrapped tensor as a list, tuple, Numpy ``ndarray``, scalar, etc.
    :type values: array_like
    :param time_dim: the index of the time dimension.
    :type time_dim: ``int``, optional
    :param size: size of the timetensor, if not give, size will be deduce from *indices*.
    :type size: list, tuple, or ``torch.Size``, optional
    :param dtype: the desired data type of the wrapped tensor (default: None, infered from ``data``).
    :type dtype: :class:`torch.dtype`, optional
    :param device: the estination device of the wrapped tensor (default: None, current device, see ``torch.set_default_tensor_type()``).
    :type device: :class:`torch.device`, optional
    :param requires_grad: Should operations been recorded by autograd for this timetensor?
    :type requires_grad: `bool`, optional

    Example:

        >>> echotorch.sparse_coo_timetensor(indices=torch.tensor([[0, 1, 1], [2, 0, 2]]), values=torch.tensor([3, 4, 5], dtype=torch.float32), size=[2, 4])
        timetensor(indices=tensor([[0, 1, 1],
                                   [2, 0, 2]]),
                   values=tensor([3., 4., 5.]),
                   size=(2, 4), nnz=3, layout=torch.sparse_coo, time_dim: 0)
    """
    # Create sparse tensor
    coo_tensor = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=size,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad
    )

    # Create TimeTensor
    return TimeTensor.new_timetensor(
        data=coo_tensor,
        time_dim=time_dim
    )
# end sparse_coo_timetensor


# As strided
def as_strided(
        input,
        size,
        stride,
        storage_offset=0,
        time_dim: Optional[int] = 0,
) -> TimeTensor:
    r"""Create a view of an existing :class:`TimeTensor` with specified ``size``, ``stride`` and ``storage_offset``.

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.as_strided.html#torch.as_strided>`__ on ``as_strided()`` for more informations.

    :param input:
    :type input:
    :param size:
    :type size:
    :param stride:
    :type stride:
    :param storage_offset:
    :type storage_offset:
    :param time_dim:
    :type time_dim:

    Example:

        >>> ...

    """
    return TimeTensor.new_timetensor(
        torch.as_strided(
            input,
            size,
            stride,
            stride,
            storage_offset
        ),
        time_dim=time_dim
    )
# end as_strided


# From Numpy
def from_numpy(
        ndarray: np.ndarray,
        time_dim: Optional[int] = 0,
) -> TimeTensor:
    r"""Creates a :class:`TimeTensor` from a ``numpy.ndarray``.

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.from_numpy.html#torch.from_numpy>`__ on ``from_numpy()`` for more informations.

    :param time_dim: Index of the time dimension.
    :type time_dim: Integer
    :param ndarray: The numpy array
    :type ndarray: ``numpy.array`` or ``numpay.ndarray``

    Examples::
        >>> x = echotorch.from_numpy(np.zeros((100, 2)), time_dim=0)
        >>> x.size()
        torch.Size([100, 2])
        >>> x.tlen
        100

    """
    return TimeTensor.new_timetensor(
        torch.from_numpy(ndarray),
        time_dim=time_dim
    )
# end from_numpy


# Returns time tensor filled with zeros
def zeros(
        *size,
        length: int,
        batch_size: Optional[Tuple[int]] = None,
        out: Optional[TimeTensor] = None,
        dtype: Optional[torch.dtype] = None,
        layout: Optional[torch.layout] = torch.strided,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False
) -> 'TimeTensor':
    r"""Returns a :class:`TimeTensor` of size ``size`` filled with 0.

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.zeros.html#torch.zeros>`__ on ``zeros()`` for more informations.

    :param size: Size
    :type size: Tuple[int]
    :param length: Length of the timeseries
    :type length: int
    :param batch_size:
    :type batch_size: tuple of ``int``
    :param out:
    :type out:
    :param dtype: :class:`TimeTensor` data type
    :type dtype: torch.dtype, optional
    :param layout: desired layout of wrapped tensor (default: ``torch.strided``)
    :type layout: torch.layout, optional
    :param device: Destination device
    :type device: torch.device, optional
    :param requires_grad: Activate gradient computation
    :type requires_grad: bool, optional

    Example::
        >>> x = echotorch.zeros((2, 2), length=100)
        >>> x.size()
        torch.Size([100, 2, 2])
        >>> x.tsize()
        torch.Size([2, 2])
        >>> x.tlen
        100
        >>> echotorch.zeros((), length=5)
        timetensor([ 0., 0., 0., 0., 0.])
    """
    return TimeTensor.new_timetensor_with_func(
        *size,
        func=torch.zeros,
        length=length,
        batch_size=batch_size,
        out=out,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
# end new_zeros


# Zeros like
def zeros_like(
        input,
        dtype: Optional[torch.dtype] = None,
        layout: Optional[torch.layout] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False,
        memory_format: Optional[torch.memory_format] = torch.preserve_format
) -> TimeTensor:
    r"""Returns a :class:`TimeTensor` filled with the scalar value 0, with the same size as ``input``.

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.zeros_like.html#torch.zeros_like>`__ on ``zeros_like()`` for more informations.

    :param input:
    :type input:
    :param dtype:
    :type dtype:
    :param layout:
    :type layout:
    :param device:
    :type device:
    :param requires_grad:
    :type requires_grad:
    :param memory_format:
    :type memory_format:

    Example:

        >>> echotorch.zeros_like()
    """
    return zeros(
        *list(input.csize()),
        length=input.tlen,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad
    )
# end zeros_like


# Returns time tensor filled with ones
def ones(
        *size,
        length: int,
        batch_size: Optional[Tuple[int]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False
) -> 'TimeTensor':
    r"""Returns a :class:`TimeTensor` of size ``size`` filled with 1.

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.ones.html#torch.ones>`__ on ``ones()`` for more informations.

    :param size: Size
    :type size: Tuple[int]
    :param length: Length of the timeseries
    :type length: int
    :param batch_size:
    :type batch_size:
    :param dtype: :class:`TimeTensor` data type
    :type dtype: torch.dtype
    :param device: Destination device
    :type device: torch.device
    :param requires_grad: Activate gradient computation
    :type requires_grad: bool

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
        length=length,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
# end ones


# Ones like
def ones_like(
        input,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False,
        memory_format=torch.preserve_format
) -> TimeTensor:
    r"""Returns a :class:`TimeTensor` filled with the scalar value 1, with the same size as ``input``.

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.ones_like.html#torch.ones_like>`__ on ``ones_like()`` for more informations.

    :param input:
    :type input:
    :param dtype:
    :type dtype:
    :param device:
    :type device:
    :param requires_grad:
    :type requires_grad:
    :param memory_format:
    :type memory_format:

    Examples:

        >>> ...
    """
    return ones(
        *list(input.csize()),
        length=input.tlen,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad
    )
# end ones_like


# Arange
def arange(
        *args,
        out: TimeTensor = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False
) -> TimeTensor:
    r"""Returns a 0-D :class:`TimeTensor` of length :math:`\ceil[\bigg]{\frac{end-start}{step}}` with values from the interval
    :math:`[start, end)` taken into common difference ``step`` beginning from *start*.

    .. note::
        **From PyTorch documentation**:
        Note that non-integer ``step`` is subject to floating point rounding errors when comparing against ``end``;
        to avoid inconsistency, we advise a small epsilon to ``end`` in such case.

        :math:`out_{i+1} = out_{i} + step`

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.arange.html#torch.arange>`__ on ``arange()`` for more informations.

    :param start: the starting value of the time related set of points (default: 0).
    :type start: Number
    :param end: the ending value for the time related set of points.
    :type end: Number
    :param step: the gap between each pair of adjacent time points (default: 1).
    :type step: Number
    :param out:
    :param dtype:
    :param device:
    :param requires_grad:

    Examples:

        >>> echotorch.tarange(0, 5)
        timetensor(tensor([0, 1, 2, 3, 4]), time_dim: 0)
        >>> echotorch.tarange(1, 4)
        timetensor(tensor([1, 2, 3]), time_dim: 0)
        >>> echotorch.tarange(1, 2.5, 0.5)
        timetensor(tensor([1.0000, 1.5000, 2.0000]), time_dim: 0)
    """
    # Get start, end, step
    if len(args) == 1:
        start = 0
        end = args[0]
        step = 1
    elif len(args) == 2:
        start = args[0]
        end = args[1]
        step = 1
    elif len(args) > 2:
        start = args[0]
        end = args[1]
        step = args[2]
    else:
        raise ValueError("At least end must be given (here nothing)")
    # end if

    # arange tensor
    ar_tensor = torch.arange(start, end, step, dtype=dtype, device=device, requires_grad=requires_grad)

    # Create timetensor
    return TimeTensor.new_timetensor(
        data=ar_tensor,
        time_dim=0
    )
# end arange


# linspace
def linspace(
        start: int,
        end: int,
        steps: float,
        out: TimeTensor = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False
) -> TimeTensor:
    r"""Create a 0-D timetensor of length ``steps`` whose values are evenly spaced from ``start`` to ``end``, inclusive.
    That is, values are:

    .. math::
        (start, start + \frac{end - start}{steps - 1}, \dots, start + (steps - 2) * \frac{end - start}{steps - 1}, end)

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.linspace.html#torch.linspace>`__ on ``linspace()`` for more informations.

    :param start: the starting value of the time related set of points.
    :type start: float
    :param end: the ending value for the time related set of points.
    :type end: float
    :param steps: size of the constructed tensor.
    :type steps: int
    :param out: the output tensor.
    :type out: Tensor, optional
    :param dtype: the data type to perform the computation in. Default: if None, uses the global default dtype (see torch.get_default_dtype()) when both start and end are real, and corresponding complex dtype when either is complex.
    :type dtype: ``torch.dtype``, optional
    :param device: the desired device of returned tensor. Default: if None, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). device will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
    :type device: ``torch.device``, optional
    :param requires_grad: if autograd should record operations on the returned tensor (default: *False*).
    :type requires_grad: ``bool``

    Example:

        >>> echotorch.linspace(3, 10, steps=5)
        timetensor(tensor([ 3.0000,  4.7500,  6.5000,  8.2500, 10.0000]), time_dim: 0)
        >>> echotorch.linspace(-10, 10, steps=5)
        timetensor(tensor([-10.,  -5.,   0.,   5.,  10.]), time_dim: 0)
        >>> echotorch.linspace(start=-10, end=10, steps=5)
        timetensor(tensor([-10.,  -5.,   0.,   5.,  10.]), time_dim: 0)
        >>> echotorch.linspace(start=-10, end=10, steps=1)
        timetensor(tensor([-10.]), time_dim: 0)
    """
    # linspace tensor
    ls_tensor = torch.linspace(start, end, steps, dtype=dtype, device=device, requires_grad=requires_grad)

    # Create timetensor
    return TimeTensor.new_timetensor(
        data=ls_tensor,
        time_dim=0
    )
# end linspace


# logspace
def logspace(
        start: int,
        end: int,
        steps: float,
        base: float = 10,
        out: TimeTensor = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False
) -> TimeTensor:
    r"""Create a 0-D timetensor of length ``steps`` whose values are evenly spaced from :math:`base^{start}` to :math:`base^{end}`,
    inclusive, on a logarithm scale with base ``base``. That is, the values are:

    .. math::
        (base^{start}, base^{\frac{end - start}{steps - 1}}, \dots, base^{start + (steps - 2) * \frac{end - start}{steps - 1}}, base^{end})

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.arange.html#torch.logspace>`__ on ``logspace()`` for more informations.

    :param start: the starting value of the time related set of points.
    :type start: float
    :param end: the ending value for the time related set of points.
    :type end: float
    :param steps: size of the constructed tensor.
    :type steps: int
    :param base:
    :type base:
    :param out: the output tensor.
    :type out: ``TimeTensor``
    :param dtype: the data type to perform the computation in. Default: if None, uses the global default dtype (see torch.get_default_dtype()) when both start and end are real, and corresponding complex dtype when either is complex.
    :type dtype: ``torch.dtype``, optional
    :param device: the desired device of returned tensor. Default: if None, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). device will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
    :type device: ``torch.device``, optional
    :param requires_grad: if autograd should record operations on the returned tensor (default: *False*).
    :type requires_grad: ``bool``
    :return: A 0-D timetensor of length ``steps`` whose values are evenly spaced from :math:`base^{start}` to :math:`base^{end}`,
    inclusive, on a logarithm scale with base ``base``.
    :rtype: ``TimeTensor``

    Example:

        >>> echotorch.logspace(start=-10, end=10, steps=5)
        timetensor(tensor([1.0000e-10, 1.0000e-05, 1.0000e+00, 1.0000e+05, 1.0000e+10]), time_dim: 0)
        >>> echotorch.logspace(start=0.1, end=1.0, steps=5)
        timetensor(tensor([ 1.2589,  2.1135,  3.5481,  5.9566, 10.0000]), time_dim: 0)
        >>> echotorch.logspace(start=0.1, end=1.0, steps=1)
        timetensor(tensor([1.2589]), time_dim: 0)
        >>> echotorch.logspace(start=2, end=2, steps=1, base=2)
        timetensor(tensor([4.]), time_dim: 0)
    """
    # linspace tensor
    ls_tensor = torch.logspace(start, end, steps, base, dtype=dtype, device=device, requires_grad=requires_grad)

    # Create timetensor
    return TimeTensor.new_timetensor(
        data=ls_tensor,
        time_dim=0
    )
# end logspace


# Returns empty tensor
def empty(
        size: Tuple[int],
        length: int,
        batch_size: Optional[Tuple[int]] = None,
        out: TimeTensor = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False
) -> 'TimeTensor':
    r"""Returns a :class:`TimeTensor` of size ``size`` and time length ``time_length`` filled with uninitialized data.

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.empty.html#torch.empty>`__ on ``empty()`` for more informations.

    :param size: Size
    :type size: Tuple[int]
    :param length: Length of the timeseries
    :type length: int
    :param batch_size:
    :type batch_size:
    :param out:
    :type out:
    :param dtype: ``TimeTensor`` data type
    :type dtype: torch.dtype
    :param device: Destination device
    :type device: torch.device
    :param requires_grad: Activate gradient computation
    :type requires_grad: bool
    :return: A :class:`TimeTensor` of size size filled with zeros
    :rtype: :class:`TimeTensor`

    Example:

        >>> echotorch.empty(2, 3, length=1, dtype=torch.int32, device = 'cuda')
        timetensor([[[1., 1., 1.],
                     [1., 1., 1.]]], device='cuda:0', dtype=torch.int32)
    """
    return TimeTensor.new_timetensor_with_func(
        size,
        func=torch.empty,
        length=length,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
# end empty


# Empty like
def empty_like(
        input,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False,
        memory_format=torch.preserve_format
) -> TimeTensor:
    r"""Returns an uninitialized :class:`TimeTensor` with the same  channel size and time dimension as ``input``.
    ``echotorch.empty_like(input)`` is equivalent to ``echotorch.empty(*list(input.csize()), time_length: input.tlen, dtype=input.dtype, layout=input.layout, device=input.device)``.

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.empty_like.html#torch.empty_like>`__ on ``empty_like()`` for more informations.

    :param input: the parameters of ``input`` will determine the parameters of the output tensor.
    :type input: ``Tensor``
    :type dtype: torch.dtype
    :param device: Destination device
    :type device: torch.device
    :param requires_grad: Activate gradient computation
    :type requires_grad: bool
    :param memory_format: the desired memory format of returned :class:`TimeTensor` (default: `torch.preserve_format`)
    :return: A tensor with the same
    :rtype: :class:`TimeTensor`

    Example:

        >>> x = echotorch.empty((2, 3), time_length=1, dtype=torch.int32, device = 'cuda')
        >>> echotorch.empty_like(x)
        timetensor([[[1., 1., 1.],
                     [1., 1., 1.]]], device='cuda:0', dtype=torch.int32)
    """
    return empty(
        *list(input.csize()),
        length=input.tlen,
        dtype=input.dtype if dtype is None else dtype,
        device=input.device if device is None else device,
        requires_grad=input.requires_grad if requires_grad is None else  requires_grad
    )
# end empty_like


# Empty strided
def empty_strided(
        size,
        stride,
        length: int,
        time_stride: int,
        batch_size: Optional[Tuple[int]] = None,
        batch_stride: Optional[Tuple[int]] = None,
        dtype: torch.device = None,
        layout: torch.layout = torch.strided,
        device: torch.device = None,
        requires_grad: bool = False,
        pin_memory: bool = False
) -> TimeTensor:
    r"""Returns a :class:`TimeTensor` filled with uninitialized data. The shape and strides of the wrapped tensor is
    defined by the argument ``size`` and ``stride``. ``echotorch.empty_strided(size, stride)`` is equivalent to
    ``echotorch.empty(*size).as_strided(size, stride)``.

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.empty_strided.html#torch.empty_strided>`__ on ``empty_strided()`` for more informations.

    :param size:
    :type size:
    :param stride:
    :type stride:
    :param length:
    :type length:
    :param time_stride:
    :type time_stride:
    :param batch_size:
    :type batch_size:
    :param batch_stride:
    :type batch_stride:
    :param dtype:
    :type dtype: torch.dtype
    :param layout:
    :type layout:
    :param device: Destination device
    :type device: torch.device
    :param requires_grad: Activate gradient computation
    :type requires_grad: bool
    :param memory_format: the desired memory format of returned :class:`TimeTensor` (default: `torch.preserve_format`)

    """
    # Data tensor
    data_tensor = torch.empty_strided(
        [length] + list(size),
        [time_stride] + list(stride),
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
        pin_memory=pin_memory
    )

    # Create timetensor
    return TimeTensor.new_timetensor(
        data=data_tensor,
        time_dim=0
    )
# end empty_strided


# Returns filled time tensor
def full(
        *size,
        fill_value: Union[int, float],
        length: int,
        batch_size: Optional[Tuple[int]] = None,
        out: Optional[TimeTensor] = None,
        dtype: Optional[torch.dtype] = None,
        layout: Optional[torch.layout] = torch.strided,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False
) -> 'TimeTensor':
    r"""Returns a TimeTensor of size size and time length time_length filled with fill_value.

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.full.html#torch.full>`__ on ``full()`` for more informations.

    :param size: Size
    :type size: Tuple[int]
    :param fill_value: the value to fill the output timetensor with.
    :type fill_value: Scalar
    :param length: Length of the timeseries
    :type length: int
    :param batch_size: Batch size
    :type batch_size: ``tuple`` of ``int``
    :param out: Output timetensor.
    :type out: :class:`TimeTensor`
    :param dtype: ``TimeTensor`` data type.
    :type dtype: torch.dtype
    :param layout: TODO: doc
    :type layout: ...
    :param device: Destination device
    :type device: torch.device
    :param requires_grad: Activate gradient computation
    :type requires_grad: bool

    Example::
        >>> x = echotorch.full(2, 2, length=100)
        >>> x.size()
        torch.Size([100, 2, 2])
        >>> x.tsize()
        torch.Size([2, 2])
        >>> x.tlen
        100
        >>> echotorch.full(fill_value=1, length=5)
        timetensor([ 1., 1., 1., 1., 1.])
    """
    # Size
    if out is not None:
        out = TimeTensor.new_timetensor_with_func(
            *size,
            func=torch.full,
            length=length,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            fill_value=fill_value,
            layout=layout
        )
        return out
    else:
        return TimeTensor.new_timetensor_with_func(
            *size,
            func=torch.full,
            length=length,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            fill_value=fill_value,
            layout=layout
        )
    # end if
# end full


# Full like
def full_like(
        input,
        fill_value: Union[int, float],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False,
        memory_format=torch.preserve_format
) -> TimeTensor:
    r"""

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.full_like.html#torch.full_like>`__ on ``full_like()`` for more informations.

    :param input:
    :type input:
    :param fill_value:
    :type fill_value:
    :param dtype:
    :type dtype:
    :param device:
    :type device:
    :param requires_grad:
    :type requires_grad:
    :param memory_format:
    :type memory_format:

    Example:

        >>> ...
    """
    pass
# end full_like


# Quantize per timetensor
def quantize_per_timetensor(
        input: TimeTensor,
        scale: float,
        zero_point: int,
        dtype: Optional[torch.dtype] = None
) -> TimeTensor:
    r"""

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html#torch.quantize_per_tensor>`__ on ``quantize_per_tensor()`` for more informations.

    :param input:
    :type input:
    :param scale:
    :type scale:
    :param zero_point:
    :type zero_point:
    :param dtype:
    :type dtype:

    Example:

        >>> ...
    """
    pass
# end quantize_per_timetensor


# Quantize per channel
def quantize_per_channel(
        input: TimeTensor,
        scales: torch.Tensor,
        zero_points: int,
        axis: int,
        dtype: Optional[torch.dtype] = None,
) -> TimeTensor:
    r"""

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.quantize_per_channel.html#torch.quantize_per_channel>`__ on ``quantize_per_channel()`` for more informations.

    :param input:
    :type input:
    :param scales:
    :type scales:
    :param zero_points:
    :type zero_points:
    :param axis:
    :type axis:
    :param dtype:
    :type dtype:

    Example:

        >>> ...
    """
    pass
# end quantize_per_channel


# Dequantize
def dequantize(
        timetensor: TimeTensor
) -> TimeTensor:
    r"""

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.dequantize.html#torch.dequantize>`__ on ``dequantize()`` for more informations.

    :param timetensor:
    :type timetensor:

    Example:

        >>> ...
    """
    pass
# end dequantize


# Complex
def complex(
        real: TimeTensor,
        imag: TimeTensor
) -> TimeTensor:
    r"""

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.complex.html#torch.complex>`__ on ``complex()`` for more informations.

    :param real:
    :type real:
    :param imag:
    :type imag:

    Example:

        >>> ...
    """
    pass
# end complex


# Polar
def polar(
        abs: TimeTensor,
        angle: TimeTensor,
        out: Optional[TimeTensor] = None,
) -> TimeTensor:
    r"""

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.polar.html#torch.polar>`__ on ``polar()`` for more informations.

    :param abs:
    :type abs:
    :param angle:
    :type angle:
    :param out:
    :type out:

    Example:

        >>> ...
    """
    pass
# end polar


# endregion CREATION_OPS


# region DISTRIBUTION_OPS


# Random time series (uniform)
def rand(
        *size,
        length: int,
        batch_size: Optional[Tuple[int]] = None,
        out: Optional[TimeTensor] = None,
        dtype: Optional[torch.dtype] = None,
        layout: Optional[torch.layout] = torch.strided,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False
) -> TimeTensor:
    r"""Returns a :class:`TimeTensor` filled with random numbers from a uniform distribution on the interval :math:`[0, 1)`.

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.rand.html#torch.rand>`__ on ``rand()`` for more informations.

    :param size: a sequence of integers defining the shape of the output timeseries. Can be a variable number of arguments or a collection like a list or tuple.
    :type size: list of integers
    :param length: length of the timeseries.
    :type length: ``int``
    :param batch_size:
    :type batch_size:
    :param out: the output tensor.
    :type out: :class:`TimeTensor`, optional
    :param out: the output tensor.
    :type out: :class:`TimeTensor`, optional
    :param dtype: the desired data type of the wrapped tensor (default: None, infered from ``data``).
    :type dtype: :class:`torch.dtype`, optional
    :param layout: the desired layout of returned TimeTensor (default: torch.strided).
    :type layout: ``torch.layout``, optional
    :param device: the estination device of the wrapped tensor (default: None, current device, see ``torch.set_default_tensor_type()``).
    :type device: :class:`torch.device`, optional
    :param requires_grad: Should operations been recorded by autograd for this timetensor?
    :type requires_grad: `bool`, optional

    Example:

        >>> echotorch.rand(2, length=10)
        timetensor([[0.5474, 0.7742],
                    [0.8091, 0.3192],
                    [0.6742, 0.3458],
                    [0.6646, 0.5652],
                    [0.4309, 0.5330],
                    [0.4052, 0.5731],
                    [0.2499, 0.1044],
                    [0.9394, 0.0862],
                    [0.2206, 0.9380],
                    [0.1908, 0.0594]], time_dim: 0)
    """
    return TimeTensor.new_timetensor_with_func(
        *size,
        func=torch.rand,
        length=length,
        batch_size=batch_size,
        out=out,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        layout=layout
    )
# end rand


# Random time series (uniform)
def randn(
        *size,
        length: int,
        batch_size: Optional[Tuple[int]] = None,
        out: Optional[TimeTensor] = None,
        dtype: Optional[torch.dtype] = None,
        layout: Optional[torch.layout] = torch.strided,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False
) -> TimeTensor:
    r"""Returns a :class:`TimeTensor` filled with random numbers from a normal distribution with mean :math:`\mu` 0 and
    a standard deviation :math:`\sigma` of 1 (standard normal distribution).

    .. math::
        out_i \sim \mathcal{N}(0, 1)

    The parameter *size* will determine the size of the *timetensor*.

    .. seealso::
        See the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.randn.html#torch.randn>`__ on ``randn()`` for more informations.

    :param size: the shape of the timetensor as a sequence of integers (list, tuple, etc).
    :type size: list of ints
    :param length: Length of the timeseries (default: 0)
    :type length: ``int``, optional
    :param batch_size:
    :type batch_size:
    :param out: the output tensor.
    :type out: ``TimeTensor``, optional
    :param dtype: the desired data type of the wrapped tensor (default: None, infered from ``data``).
    :type dtype: :class:`torch.dtype`, optional
    :param layout: desired layout of the wrapped Tensor (default: ``torch.strided``).
    :type layout: ``torch.layout``, optional
    :param device: the estination device of the wrapped tensor (default: None, current device, see ``torch.set_default_tensor_type()``).
    :type device: :class:`torch.device`, optional
    :param requires_grad: Should operations been recorded by autograd for this timetensor?
    :type requires_grad: `bool`, optional

    Example:

        >>> x = echotorch.randn(length=10)
        >>> x
        timetensor([ 0.2610,  0.4589,  0.1833, -0.1209, -0.0103,  1.1757,  0.9236, -0.6117, 0.7906, -0.1704], time_dim: 0)
        >>> x.size()
        torch.Size([10])
        >>> x.tlen
        10
    """
    return TimeTensor.new_timetensor_with_func(
        *size,
        func=torch.randn,
        length=length,
        batch_size=batch_size,
        out=out,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        layout=layout
    )
# end randn


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
def cat(
        tensors: Tuple[TimeTensor],
        dim: int = 0
) -> Union[TimeTensor, Any]:
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
