# -*- coding: utf-8 -*-
#
# File : echotorch/timetensor.py
# Description : A special tensor with a time dimension
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
from typing import Optional, Tuple, Union, List, Callable, Any
import torch
import numpy as np

# EchoTorch imports
import echotorch
from .base_tensors import BaseTensor


# Error
ERROR_TENSOR_TO_SMALL = "Time dimension does not exists in the data tensor " \
                        "(time dim at {}, {} dimension in tensor). The minimum tensor size " \
                        "is {}"
ERROR_TIME_LENGTHS_TOO_BIG = "There is time lengths which are bigger than the actual tensor data"
ERROR_WRONG_TIME_LENGTHS_SIZES = "The sizes of the time lengths tensor should be {}"
ERROR_TIME_DIM_NEGATIVE = "The index of the time-dimension cannot be negative"


# Torch ops which can be directly converted to timetensors
TORCH_OPS_DIRECT = [
    'cat', 'chunk', 'column_stack', 'dsplit', 'gather', 'hsplit', 'hstack', 'index_select', 'narrow', 'scatter',
    'scatter_add', "split"
]


# region TIMETENSOR

# TimeTensor
def check_time_lengths(
        time_len: int,
        time_lengths: Optional[torch.LongTensor],
        batch_sizes: torch.Size
):
    r"""Check time lengths

    :param time_lengths:
    :param batch_sizes:
    :return:
    """
    # Check that the given lengths tensor has the right
    # dimensions
    if time_lengths.size() != batch_sizes:
        raise ValueError(ERROR_WRONG_TIME_LENGTHS_SIZES.format(batch_sizes))
    # end if

    # Check that all lengths are not bigger
    # than the actual time-tensor
    if torch.any(time_lengths > time_len):
        raise ValueError(ERROR_TIME_LENGTHS_TOO_BIG)
    # end if

    return True
# end check_time_lengths


# TimeTensor
class TimeTensor(BaseTensor):
    r"""A  special tensor with a time dimension.
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'TimeTensor'],
            time_dim: Optional[int] = 0
    ) -> None:
        r"""TimeTensor constructor

        :param data: The data in a torch tensor to transform to timetensor.
        :param time_dim: The position of the time dimension.
        """
        # Copy if already a timetensor
        # transform otherwise
        if type(data) is TimeTensor:
            tensor_data = data.tensor
        else:
            tensor_data = data
        # end if

        # The tensor must have enough dimension
        # for the time dimension
        if tensor_data.ndim < time_dim + 1:
            # Error
            raise ValueError(
                ERROR_TENSOR_TO_SMALL.format(time_dim, tensor_data.ndim, time_dim + 1)
            )
        # end if

        # Set tensor and time index
        self._tensor = tensor_data
        self._time_dim = time_dim
    # end __init__

    # endregion CONSTRUCTORS

    # region PROPERTIES

    # Time dimension (getter)
    @property
    def time_dim(self) -> int:
        r"""Get the index of the time dimension.

        :return: The index of the time dimension.
        :rtype: ``ìnt``
        """
        return self._time_dim
    # end time_dim

    # Time dimension (setter)
    @time_dim.setter
    def time_dim(
            self,
            value: int
    ) -> None:
        r"""Set the index of the time dimension if valid.

        :param value: New index of the time dimension.
        :type value: ``ìnt``
        """
        # Check time dim is valid
        if value >= self.tensor.ndim:
            # Error
            raise ValueError(ERROR_TENSOR_TO_SMALL.format(value, self._tensor.ndim))
        elif value < 0:
            raise ValueError(ERROR_TIME_DIM_NEGATIVE)
        # end if

        # Set new time dim
        self._time_dim = value
    # end time_dim

    # Time length
    @property
    def tlen(self) -> int:
        r"""Returns the length of the time dimension.

        :return: the length of the time dimension.
        :rtype: ``int``
        """
        return self._tensor.size()[self._time_dim]
    # end tlen

    # Number of channel dimensions
    @property
    def cdim(self) -> int:
        r"""Number of channel dimensions.

        :return: the number of channel dimensions.
        :rtype: ``ìnt``
        """
        return self._tensor.ndim - self._time_dim - 1
    # end cdim

    # Number of batch dimensions
    @property
    def bdim(self) -> int:
        r"""Number of batch dimensions.

        :return: the number of batch dimensions.
        :rtype: ``ìnt``
        """
        return self._tensor.ndim - self.cdim - 1
    # end bdim

    # endregion PROPERTIES

    # region PUBLIC

    # Size of channel dimensions
    def csize(self) -> torch.Size:
        r"""Size of channel dimensions.
        """
        if self._time_dim != self._tensor.ndim - 1:
            tensor_size = self._tensor.size()
            return tensor_size[self.time_dim+1:]
        else:
            return torch.Size([])
        # end if
    # end csize

    # Size of batch dimensions
    def bsize(self) -> torch.Size:
        r"""Size of batch dimensions.
        """
        if self._time_dim == 0:
            return torch.Size([])
        else:
            tensor_size = self._tensor.size()
            return tensor_size[:self._time_dim]
        # end if
    # end bsize

    # Number of channel elements
    def numelc(self):
        r"""Returns the number of elements in the channel dimensions.
        """
        # Multiply sizes
        num_el = 1
        for c_size in list(self.csize()):
            num_el *= c_size
        # end for
        return num_el
    # end numelc

    # Number of batch elements
    def numelb(self):
        r"""Returns the number of elements in the batch dimensions.
        """
        # Multiply sizes
        num_el = 1
        for b_size in list(self.bsize()):
            num_el *= b_size
        # end for
        return num_el
    # end numelb

    # region CAST

    # To
    def to(self, *args, **kwargs) -> 'TimeTensor':
        r"""Performs TimeTensor dtype and/or device concersion. A ``torch.dtype`` and ``torch.device`` are inferred
        from the arguments of ``self.to(*args, **kwargs)

        .. note::
            From PyTorch documentation: if the ``self`` TimeTensor already has the correct ``torch.dtype`` and
            ``torch.device``, then ``self`` is returned. Otherwise, the returned timetensor is a copy of ``self``
            with the desired ``torch.dtype`` and ``torch.device``.

        Example::
            >>> ttensor = echotorch.randn(2, length=20)
            >>> ttensor.to(torch.float64)

        """
        # New tensor
        ntensor = self._tensor.to(*args, **kwargs)

        # Same tensor?
        if self._tensor == ntensor:
            return self
        else:
            return TimeTensor(
                ntensor,
                time_dim=self._time_dim
            )
        # end if
    # end to

    # endregion CAST

    # Indexing time tensor
    def indexing_timetensor(
            self,
            item
    ) -> 'TimeTensor':
        r"""Return a view of a :class:`TimeTensor` according to an indexing item.

        :param item: Data item to recover.
        :rtype: :class:`TimeTensor`
        """
        return TimeTensor(
            self._tensor[item],
            time_dim=self._time_dim
        )
    # end indexing_timetensor

    # endregion PUBLIC

    # region TORCH_FUNCTION

    # After unsqueeze
    def after_unsqueeze(
            self,
            func_output: Any,
            input: Any,
            dim: int
    ) -> 'TimeTensor':
        r"""After unsqueeze

        :param func_output: The output of the torch.unsqueeze function.
        :type func_output:
        :param dim: The request dimension from unsqueeze.
        :type dim:
        :return: The computed output.
        :rtype:
        """
        if dim <= self.time_dim:
            return TimeTensor(
                func_output,
                time_dim=self._time_dim+1
            )
        else:
            return TimeTensor(
                func_output,
                time_dim=self._time_dim
            )
        # end if
    # end after_unsqueeze

    # After dstack
    def after_dstack(
            self,
            func_output: Any,
            *ops_inputs,
            dim: int = 0
    ) -> 'TimeTensor':
        r"""After :func:`torch.dstack`.
        """
        # 0-D timeseries
        if self.ndim == 1:
            return TimeTensor(
                data=func_output,
                time_dim=1
            )
        else:
            return TimeTensor(
                data=func_output,
                time_dim=self._time_dim
            )
        # end if
    # end after_dstack

    # After movedim
    def after_movedim(
            self,
            func_output: Any,
            ops_input,
            source,
            destination
    ) -> 'TimeTensor':
        r"""After :func:`torch.movedim` we change index of time dimension of concerned.
        """
        # Keep time dim
        new_time_dim = self.time_dim

        # New time dim if in dest or source
        if source == self.time_dim:
            new_time_dim = destination
        elif destination == self.time_dim:
            new_time_dim = source
        # end if

        # New timetensor
        return TimeTensor(
            data=func_output,
            time_dim=new_time_dim
        )
    # end after_movedim

    # After squeeze
    def after_squeeze(
            self,
            func_output,
            input,
            dim
    ) -> Union['TimeTensor', torch.Tensor]:
        r"""
        """
        if dim is None:
            if self.tlen == 1:
                return func_output
            else:
                # How many dim at one before time dim?
                removed_dim = torch.sum(torch.tensor(self.size()[:self.time_dim]) == 1)

                # Return with modified time dim
                return TimeTensor(
                    data=func_output,
                    time_dim=self.time_dim - removed_dim
                )
            # end if
        else:
            # Time dim targeted
            if dim == self.time_dim and self.tlen == 1:
                return func_output
            # end if

            # If dim removed and before time dim
            if self.size()[dim] == 1 and dim < self.time_dim:
                return TimeTensor(
                    data=func_output,
                    time_dim=self.time_dim - 1
                )
            else:
                return TimeTensor(
                    data=func_output,
                    time_dim=self.time_dim
                )
            # end if
        # end if
    # end after_squeeze

    # After atleast_3d
    def after_atleast_3d(
            self,
            func_output: Any,
            *ops_inputs,
            dim: int = 0
    ) -> 'TimeTensor':
        r"""After :func:`torch.atleast_3d`.
        """
        # 0-D timeseries
        if self.ndim == 1:
            return TimeTensor(
                data=func_output,
                time_dim=1
            )
        else:
            return TimeTensor(
                data=func_output,
                time_dim=self._time_dim
            )
        # end if
    # end after_atleast_3d

    # Transform to timetensor
    def transform_to_timetensors(
            self,
            tensors: Any
    ):
        r"""Transform :class:`torch.Tensor` to :class:`TimeTensor`.
        """
        return [echotorch.as_timetensor(o, time_dim=self.time_dim) for o in tensors]
    # end transform_to_timetensors

    # Convert to timetensor
    def convert_to_timetensor(self, func_ret):
        r"""Convert to timetensor.
        """
        if isinstance(func_ret, torch.Tensor):
            return TimeTensor(
                data=func_ret,
                time_dim=self.time_dim
            )
        elif isinstance(func_ret, list):
            return self.transform_to_timetensors(func_ret)
        elif isinstance(func_ret, tuple):
            return tuple(self.transform_to_timetensors(func_ret))
        else:
            return func_ret
        # end if
    # end convert_to_timetensor

    # Check that all timetensors have the right time dimension index
    def check_time_dim(
            self,
            tensors: Any
    ) -> None:
        r"""Check that all timetensors have the right time dimension index.
        """
        for tensor in tensors:
            if isinstance(tensor, TimeTensor) and self.time_dim != tensor.time_dim:
                raise RuntimeError(
                    "Expected timetensors with the same time dimension index, got {} and {}".format(
                        self.time_dim,
                        tensor.time_dim
                    )
                )
            # end if
        # end for
    # end check_time_dim

    # After mm (matrix multiplication)
    def mm(
            self,
            func_output: Any,
            m1,
            m2
    ) -> Union['TimeTensor', torch.Tensor]:
        r"""After mm (matrix multiplication)

        :param m1: first tensor.
        :type m1: :class:`TimeTensor` or ``torch.Tensor``
        :param m2: second tensor.
        :type m2: :class:`TimeTensor` or ``torch.Tensor``
        """
        return func_output
    # end mm

    # Transpose
    def t(self) -> 'TimeTensor':
        r"""Expects the timetensor to be <= 1-D timetensor and transposes dimensions 0 and 1.

        If time dimension is in position 0, then it is switched to 1, and vice versa.

        TODO: complet doc
        """
        if self.ndim == 2:
            return TimeTensor(
                data=self._tensor.t(),
                time_dim=1-self._time_dim
            )
        elif self.ndim < 2:
            return self
        else:
            # Inverse time dim if targeted
            if self._time_dim in [0, 1]:
                return TimeTensor(
                    data=self._tensor.t(),
                    time_dim=1 - self._time_dim
                )
            else:
                return TimeTensor(
                    data=self._tensor.t(),
                    time_dim=self._time_dim
                )
            # end if
        # end if
    # end t

    # As strided
    def as_strided(
            self,
            size,
            stride,
            storage_offset=0,
            time_dim=None
    ) -> 'TimeTensor':
        r"""TODO: document

        :param size:
        :param stride:
        :param storage_offset:
        :return:
        :rtype:
        """
        # Strided tensor
        data_tensor = self._tensor.as_strided(size, stride, storage_offset)

        # Time dim still present
        if len(size) >= self._time_dim + 1:
            # Return timetensor
            return TimeTensor.new_timetensor(
                data=data_tensor,
                time_dim=self._time_dim if time_dim is None else time_dim
            )
        elif time_dim is not None:
            pass
        else:
            return data_tensor
        # end if
    # end as_strided

    # Torch functions
    def __torch_function__(
            self,
            func,
            types,
            args=(),
            kwargs=None
    ):
        """
        Torch functions
        """
        # Dict if None
        if kwargs is None:
            kwargs = {}

        # end if

        # Convert timetensor to tensors
        def convert(args):
            if type(args) is TimeTensor:
                return args.tensor
            elif type(args) is tuple:
                return tuple([convert(a) for a in args])
            elif type(args) is list:
                return [convert(a) for a in args]
            else:
                return args
            # end if

        # end convert
        # print(func.__name__)
        # print(args)
        # print(kwargs)
        # Validate ops inputs
        if hasattr(self, 'validate_' + func.__name__): getattr(self, 'validate_' + func.__name__)(*args, **kwargs)

        # Before callback
        if hasattr(self, 'before_' + func.__name__): args = getattr(self, 'before_' + func.__name__)(*args, **kwargs)

        # Get the tensor in the arguments
        conv_args = [convert(a) for a in args]

        # Middle callback
        if hasattr(self, 'middle_' + func.__name__): args = getattr(self, 'middle_' + func.__name__)(*args, **kwargs)

        # Execute function
        ret = func(*conv_args, **kwargs)

        # If output can be directly converted to timetensor
        if func.__name__ in TORCH_OPS_DIRECT:
            ret = self.convert_to_timetensor(ret)
        # end if
        # print("FUNC RET: {}".format(type(ret)))
        # print("")
        # Create TimeTensor and returns or returns directly
        if hasattr(self, 'after_' + func.__name__):
            return getattr(self, 'after_' + func.__name__)(ret, *args, **kwargs)
        else:
            return ret
        # end if
    # end __torch_function__

    # endregion TORCH_FUNCTION

    # region OVERRIDE

    # Get item
    def __getitem__(self, item) -> Union['TimeTensor', torch.Tensor]:
        """
        Get data in the tensor
        """
        # Multiple indices
        if type(item) is tuple:
            # If time dim is in
            if len(item) > self._time_dim:
                # Selection or slice?
                if type(item[self._time_dim]) in [slice, list]:
                    return self.indexing_timetensor(item)
                else:
                    return self._tensor[item]
                # end if
            else:
                return self.indexing_timetensor(item)
            # end if
        elif type(item) in [slice, list]:
            return self.indexing_timetensor(item)
        else:
            # Time selection?
            if self._time_dim == 0:
                return self._tensor[item]
            else:
                return self.indexing_timetensor(item)
        # end if
    # end __getitem__

    # Set item
    def __setitem__(self, key, value) -> None:
        """
        Set data in the tensor
        """
        self._tensor[key] = value
    # end __setitem__

    # Length
    def __len__(self) -> int:
        """
        Time length of the time series
        """
        return self.tlen
    # end __len__

    # Get representation
    def __repr__(self) -> str:
        r"""Get a string representation

        :return: ``TimeTensor`` representation.
        :rtype: ``str``
        """
        tensor_desc = self._tensor.__repr__()
        tensor_desc = tensor_desc[7:-1]
        return "timetensor({}, time_dim: {})".format(tensor_desc, self._time_dim)
    # end __repr__

    # Are two time-tensors equivalent
    def __eq__(
            self,
            other: 'TimeTensor'
    ) -> bool:
        r"""Are two time-tensors equivalent?

        :param other: The other time-tensor
        :type other: ``TimeTensor``
        :return: True of False if the two time-tensors are equivalent
        :rtype: ``bool``

        """
        return super(TimeTensor, self).__eq__(other) and self.time_dim == other.time_dim
    # end __eq__

    # endregion OVERRIDE

    # region STATIC

    # Returns a new TimeTensor with data as the tensor data.
    @classmethod
    def new_timetensor(
            cls,
            data: Union[torch.Tensor, 'TimeTensor'],
            time_dim: Optional[int] = 0
    ) -> 'TimeTensor':
        """
        Returns a new TimeTensor with data as the tensor data.
        @param data:
        @param time_lengths:
        @param time_dim:
        @param copy_data:
        @return:
        """
        return TimeTensor(
            data,
            time_dim=time_dim
        )
    # end new_timetensor

    # Returns new time tensor with a specific function
    @classmethod
    def new_timetensor_with_func(
            cls,
            *size: Tuple[int],
            func: Callable,
            length: int,
            batch_size: Optional[Tuple[int]] = None,
            out: Optional['TimeTensor'] = None,
            **kwargs
    ) -> 'TimeTensor':
        r"""Returns a new :class:`TimeTensor` with a specific function to generate the data.

        :param func:
        :type func:
        :param size:
        :type size:
        :param length:
        :type length:
        :param batch_size:
        :type batch_size:
        :param out:
        :type out:
        """
        # Batch size
        batch_size = tuple() if batch_size is None else batch_size

        # Time dim
        time_dim = len(batch_size)

        # Total size
        tt_size = list(batch_size) + [length] + list(size)

        # Output object
        out = out.tensor if isinstance(out, TimeTensor) else out

        # Out mode
        if out is None:
            # Create TimeTensor
            return TimeTensor(
                data=func(tuple(tt_size), **kwargs),
                time_dim=time_dim
            )
        else:
            # Call function
            func(tuple(tt_size), out=out.tensor, **kwargs)
            out.time_dim = time_dim
            return out
        # end if
    # end new_timetensor_with_func

    # endregion STATIC

# end TimeTensor

# endregion TIMETENSOR


# region VARIANTS

# Float time tensor
class FloatTimeTensor(TimeTensor):
    r"""Float time tensor.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'TimeTensor'],
            time_dim: Optional[int] = 0
    ) -> None:
        r"""Float TimeTensor constructor

        :param data: The data in a torch tensor to transform to timetensor.
        :param time_dim: The position of the time dimension.
        """
        # Super call
        super(FloatTimeTensor, self).__init__(
            self,
            data,
            time_dim=time_dim
        )

        # Transform type
        self.float()
    # end __init__

# end FloatTimeTensor


# Double time tensor
class DoubleTimeTensor(TimeTensor):
    r"""Double time tensor.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'TimeTensor'],
            time_dim: Optional[int] = 0
    ) -> None:
        r"""Double TimeTensor constructor

        :param data: The data in a torch tensor to transform to timetensor.
        :param time_dim: The position of the time dimension.
        """
        # Super call
        super(DoubleTimeTensor, self).__init__(
            self,
            data,
            time_dim=time_dim
        )

        # Cast data
        self.double()
    # end __init__

# end DoubleTimeTensor


# Half time tensor
class HalfTimeTensor(TimeTensor):
    r"""Half time tensor.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'TimeTensor'],
            time_dim: Optional[int] = 0
    ) -> None:
        r"""Half TimeTensor constructor

        :param data: The data in a torch tensor to transform to timetensor.
        :param time_dim: The position of the time dimension.
        """
        # Super call
        super(HalfTimeTensor, self).__init__(
            self,
            data,
            time_dim=time_dim
        )

        # Cast data
        self.halt()
    # end __init__

# end HalfTimeTensor


# 16-bit floating point 2 time tensor
class BFloat16Tensor(TimeTensor):
    r"""16-bit floating point 2 time tensor.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'TimeTensor'],
            time_dim: Optional[int] = 0
    ) -> None:
        r"""16-bit TimeTensor constructor

        :param data: The data in a torch tensor to transform to timetensor.
        :param time_dim: The position of the time dimension.
        """
        # Super call
        super(BFloat16Tensor, self).__init__(
            self,
            data,
            time_dim=time_dim
        )

        # Cast
        self.bfloat16()
    # end __init__

# end BFloat16Tensor


# 8-bit integer (unsigned) time tensor
class ByteTimeTensor(TimeTensor):
    r"""8-bit integer (unsigned) time tensor.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'TimeTensor'],
            time_dim: Optional[int] = 0
    ) -> None:
        r"""8-bit integer (unsigned) TimeTensor constructor

        :param data: The data in a torch tensor to transform to timetensor.
        :param time_dim: The position of the time dimension.
        """
        # Super call
        super(ByteTimeTensor, self).__init__(
            self,
            data,
            time_dim=time_dim
        )

        # Cast
        self.byte()
    # end __init__

# end ByteTimeTensor


# 8-bit integer (signed) time tensor
class CharTimeTensor(TimeTensor):
    r"""8-bit integer (unsigned) time tensor.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'TimeTensor'],
            time_dim: Optional[int] = 0
    ) -> None:
        r"""CharTimeTensor constructor

        :param data: The data in a torch tensor to transform to timetensor.
        :param time_dim: The position of the time dimension.
        """
        # Super call
        super(CharTimeTensor, self).__init__(
            self,
            data,
            time_dim=time_dim
        )

        # Cast
        self.char()
    # end __init__

# end CharTimeTensor


# Boolean time tensor
class BooleanTimeTensor(TimeTensor):
    r"""Boolean time tensor.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'TimeTensor'],
            time_dim: Optional[int] = 0
    ) -> None:
        r"""BooleanTimeTensor constructor

        :param data: The data in a torch tensor to transform to timetensor.
        :param time_dim: The position of the time dimension.
        """
        # Super call
        super(BooleanTimeTensor, self).__init__(
            self,
            data,
            time_dim=time_dim
        )

        # Cast
        self.bool()
    # end __init__

# end CharTimeTensor


# endregion VARIANTS
