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
from .base_tensors import BaseTensor


# Error
ERROR_TENSOR_TO_SMALL = "Time dimension does not exists in the data tensor " \
                        "(time dim at {}, {} dimension in tensor). The minimum tensor size " \
                        "is {}"
ERROR_TIME_LENGTHS_TOO_BIG = "There is time lengths which are bigger than the actual tensor data"
ERROR_WRONG_TIME_LENGTHS_SIZES = "The sizes of the time lengths tensor should be {}"
ERROR_TIME_DIM_NEGATIVE = "The index of the time-dimension cannot be negative"


# region TIMETENSOR

# TimeTensor
def check_time_lengths(
        time_len: int,
        time_lengths: Optional[torch.LongTensor],
        batch_sizes: torch.Size
):
    """
    Check time lengths
    @param time_lengths:
    @param batch_sizes:
    @return:
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


class TimeTensor(BaseTensor):
    """
    A  special tensor with a time and a batch dimension
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'TimeTensor'],
            time_lengths: Optional[torch.LongTensor] = None,
            time_dim: Optional[int] = 0
    ) -> None:
        r"""TimeTensor constructor

        Args:
            data: The data in a torch tensor to transform to timetensor.
            time_lengths: Lengths of each timeseries.
            time_dim: The position of the time dimension.
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
            raise ValueError(ERROR_TENSOR_TO_SMALL.format(time_dim, tensor_data.ndim, time_dim + 1))
        # end if

        # Batch sizes and time length
        time_len = tensor_data.size(time_dim)
        batch_sizes = tensor_data.size()[:time_dim]

        # Set tensor and time index
        self._tensor = tensor_data
        self._time_dim = time_dim

        # Compute lengths if not given or check given ones
        if time_lengths is None:
            self._time_lengths = torch.full(batch_sizes, fill_value=time_len).long()
        else:
            # Check time lengths
            check_time_lengths(time_len, time_lengths, batch_sizes)

            # Lengths
            self._time_lengths = time_lengths
        # end if
    # end __init__

    # endregion CONSTRUCTORS

    # region PROPERTIES

    # Get timetensor device
    @property
    def device(self) -> torch.device:
        """
        Get timetensor device
        @return: Device
        """
        return self._tensor.device
    # end device

    # Get timetensor gradient policy
    @property
    def requires_grad(self) -> bool:
        """
        Get timetensor gradient policy
        @return: True/False
        """
        return self._tensor.requires_grad
    # end requires_grad

    # Set timetensor gradient policy
    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """
        Set timetensor gradient policy
        @param value: Boolean value
        """
        self._tensor.requires_grad = value
    # end requires_grad

    # Get tensor
    @property
    def tensor(self) -> torch.Tensor:
        """
        Get the original tensor
        """
        return self._tensor
    # end tensor

    # Time dimension (getter)
    @property
    def time_dim(self) -> int:
        """
        Get the time dimension
        """
        return self._time_dim
    # end time_dim

    # Time dimension (setter)
    @time_dim.setter
    def time_dim(
            self,
            value: int
    ) -> None:
        """
        Set the time dimension if valid
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
        """
        Time length
        """
        return self._tensor.size()[self._time_dim]
    # end tlen

    # Time lengths
    @property
    def tlens(self) -> torch.LongTensor:
        """
        Time lengths
        @return: Time lengths
        """
        return self._time_lengths
    # end tlens

    # Set time lengths
    @tlens.setter
    def tlens(
            self,
            value: torch.LongTensor
    ) -> None:
        """
        Set time lengths
        @param value:
        @return:
        """
        # Check time lengths
        check_time_lengths(self.tlen, value, self.bsize())

        # Set
        self._time_lengths = value
    # end tlens

    # Number of dimension
    @property
    def ndim(self) -> int:
        """
        Number of dimension
        """
        return self._tensor.ndim
    # end ndim

    # Number of channel dimensions
    @property
    def cdim(self) -> int:
        """
        Number of channel dimensions
        """
        return self._tensor.ndim - self._time_dim - 1
    # end cdim

    # Number of batch dimensions
    @property
    def bdim(self) -> int:
        """
        Number of batch dimensions
        """
        return self._tensor.ndim - self.cdim - 1
    # end bdim

    # Data type
    @property
    def dtype(self) -> torch.dtype:
        """
        Get the tensor data type
        """
        return self._tensor.dtype
    # end dtype

    # Is on CUDA device
    @property
    def is_cuda(self) -> float:
        """
        Is True if the Tensor is stored on the GPU, False otherwise.
        """
        return self._tensor.is_cuda
    # end is_cuda

    # endregion PROPERTIES

    # region PUBLIC

    # Size
    def size(self) -> torch.Size:
        """
        Size
        """
        return self._tensor.size()
    # end size

    # Size of channel dimensions
    def csize(self) -> torch.Size:
        """
        Size of channel dimensions
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
        """
        Size of batch dimensions
        """
        if self._time_dim == 0:
            return torch.Size([])
        else:
            tensor_size = self._tensor.size()
            return tensor_size[:self._time_dim]
        # end if
    # end bsize

    # Long
    def long(self) -> 'TimeTensor':
        r"""To long timetensor (no copy)

        Returns: The timetensor with data casted to long

        """
        self._tensor = self._tensor.long()
        return self
    # end long

    # Float
    def float(self) -> 'TimeTensor':
        r"""To float32 timetensor (no copy)

        Returns: The timetensor with data coasted to float32

        """
        self._tensor = self._tensor.float()
        return self
    # end float

    # Complex
    def complex(self) -> 'TimeTensor':
        r"""To complex timetensor (no copy)

        Returns: The timetensor with data casted to complex

        """
        self._tensor = self._tensor.complex()
        return self
    # end complex

    # To float16 timetensor
    def half(self) -> 'TimeTensor':
        r"""To float16 timetensor (no copy)

        Returns: The timetensor with data casted to float16

        """
        self._tensor = self._tensor.half()
        return self
    # end half

    # To
    def to(self, *args, **kwargs) -> 'TimeTensor':
        r"""Performs TimeTensor dtype and/or device concersion. A ``torch.dtype`` and ``torch.device`` are inferred
        from the arguments of ``self.to(*args, **kwargs)

        .. note::
            From PyTorch documentation: if the ``self`` TimeTensor already has the correct ``torch.dtype`` and
            ``torch.device``, then ``self`` is returned. Otherwise, the returned timetensor is a copy of ``self``
            with the desired ``torch.dtype`` and ``torch.device``.

        Args:
            *args:
            **kwargs:

        Example::
            >>> ttensor = echotorch.randn((2,), time_lengths=20)
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
                time_lengths=self._time_lengths,
                time_dim=self._time_dim
            )
        # end if
    # end to

    # Indexing time tensor
    def indexing_timetensor(
            self,
            item
    ) -> 'TimeTensor':
        """
        Return a view of a timetensor according to an indexing item
        @param item:
        @return: Timetensor
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
            dim: int
    ) -> 'TimeTensor':
        """
        After unsqueeze
        @param func_output: The output of the torch.unsqueeze function
        @param dim: The request dimension from unsqueeze
        @return: The computed output
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

    # After cat
    def after_cat(
            self,
            func_output: Any,
            *args,
            **kwargs
    ) -> 'TimeTensor':
        r"""

        Args:
            func_output:
            *args:
            **kwargs:

        Returns:

        """
        print("func_output: {}".format(func_output))
        print("args: {}".format(args))
        print("kwargs: {}".format(kwargs))
        return TimeTensor(
            data=func_output,
            time_dim=self._time_dim
        )
    # end after cat

    # endregion TORCH_FUNCTION

    # region OVERRIDE

    # To numpy
    def numpy(self) -> np.ndarray:
        r"""To Numpy array

        """
        return self._tensor.numpy()
    # end numpy

    # To CUDA device
    def cuda(
            self,
            **kwargs
    ) -> 'TimeTensor':
        """
        To CUDA device
        @return:
        """
        self._tensor = self._tensor.cuda(**kwargs)
        return self
    # end cuda

    # To CPU device
    def cpu(
            self,
            **kwargs
    ) -> 'TimeTensor':
        """
        To CPU devices
        @param kwargs:
        @return:
        """
        self._tensor = self._tensor.cpu(**kwargs)
        return self
    # end cpu

    # To
    def to(
            self,
            *args,
            **kwargs
    ) -> 'TimeTensor':
        """
        Transfer to device
        """
        self._tensor.to(*args, **kwargs)
        return self
    # end to

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
                if type(item[self._time_dim]) is slice:
                    return self.indexing_timetensor(item)
                else:
                    return self._tensor[item]
                # end if
            else:
                pass
            # end if
        elif type(item) is slice:
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
        """
        Get a string representation
        """
        return "timetensor({}, time_dim: {})".format(self._tensor, self._time_dim)
    # end __repr__

    # Are two time-tensors equivalent
    def __eq__(
            self,
            other: 'TimeTensor'
    ) -> bool:
        """
        Are two time-tensors equivalent?
        @param other: The other time-tensor
        @return: True of False if the two time-tensors are equivalent
        """
        return super(TimeTensor, self).__eq__(other) and self.time_dim == other.time_dim
    # end __eq__

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

        # Before callback
        if hasattr(self, 'before_' + func.__name__): args = getattr(self, 'before_' + func.__name__)(*args, **kwargs)

        # Get the tensor in the arguments
        args = [convert(a) for a in args]

        # Middle callback
        if hasattr(self, 'middle_' + func.__name__): args = getattr(self, 'middle_' + func.__name__)(*args, **kwargs)

        # Execute function
        ret = func(*args, **kwargs)

        # Create TimeTensor and returns or returns directly
        if hasattr(self, 'after_' + func.__name__):
            return getattr(self, 'after_' + func.__name__)(ret, **kwargs)
        elif isinstance(ret, torch.Tensor):
            # Return a new time tensor
            return TimeTensor(
                ret,
                time_dim=self._time_dim
            )
        else:
            return ret
        # end if
    # end __torch_function__

    # endregion OVERRIDE

    # region STATIC

    # Returns a new TimeTensor with data as the tensor data.
    @classmethod
    def new_timetensor(
            cls,
            data: Union[torch.Tensor, 'TimeTensor'],
            time_lengths: Optional[torch.LongTensor] = None,
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
            time_lengths=time_lengths,
            time_dim=time_dim
        )
    # end new_timetensor

    # Returns new time tensor with a specific function
    @classmethod
    def new_timetensor_with_func(
            cls,
            *size: Tuple[int],
            func: Callable,
            time_length: Union[int, torch.LongTensor],
            **kwargs
    ) -> 'TimeTensor':
        """
        Returns a new time tensor with a specific function to generate the data.
        @param func:
        @param size:
        @param time_length:
        """
        # Size
        if type(time_length) is int:
            tt_size = [time_length] + list(size)
            t_lens = None
            time_dim = 0
        else:
            tt_size = list(time_length.size()) + [torch.max(time_length).item()] + list(size)
            t_lens = time_length
            time_dim = time_length.ndim
        # end if

        # Create TimeTensor
        return TimeTensor(
            data=func(tuple(tt_size), **kwargs),
            time_dim=time_dim,
            time_lengths=t_lens
        )
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
            time_lengths: Optional[torch.LongTensor] = None,
            time_dim: Optional[int] = 0
    ) -> None:
        r"""Float TimeTensor constructor

        Args:
            data: The data in a torch tensor to transform to timetensor.
            time_lengths: Lengths of each timeseries.
            time_dim: The position of the time dimension.
        """
        # Super call
        super(FloatTimeTensor, self).__init__(
            self,
            data,
            time_lengths=time_lengths,
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
            time_lengths: Optional[torch.LongTensor] = None,
            time_dim: Optional[int] = 0
    ) -> None:
        r"""Double TimeTensor constructor

        Args:
            data: The data in a torch tensor to transform to timetensor.
            time_lengths: Lengths of each timeseries.
            time_dim: The position of the time dimension.
        """
        # Super call
        super(DoubleTimeTensor, self).__init__(
            self,
            data,
            time_lengths=time_lengths,
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
            time_lengths: Optional[torch.LongTensor] = None,
            time_dim: Optional[int] = 0
    ) -> None:
        r"""Half TimeTensor constructor

        Args:
            data: The data in a torch tensor to transform to timetensor.
            time_lengths: Lengths of each timeseries.
            time_dim: The position of the time dimension.
        """
        # Super call
        super(HalfTimeTensor, self).__init__(
            self,
            data,
            time_lengths=time_lengths,
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
            time_lengths: Optional[torch.LongTensor] = None,
            time_dim: Optional[int] = 0
    ) -> None:
        r"""16-bit TimeTensor constructor

        Args:
            data: The data in a torch tensor to transform to timetensor.
            time_lengths: Lengths of each timeseries.
            time_dim: The position of the time dimension.
        """
        # Super call
        super(BFloat16Tensor, self).__init__(
            self,
            data,
            time_lengths=time_lengths,
            time_dim=time_dim
        )
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
            time_lengths: Optional[torch.LongTensor] = None,
            time_dim: Optional[int] = 0
    ) -> None:
        r"""8-bit integer (unsigned) TimeTensor constructor

        Args:
            data: The data in a torch tensor to transform to timetensor.
            time_lengths: Lengths of each timeseries.
            time_dim: The position of the time dimension.
        """
        # Super call
        super(ByteTimeTensor, self).__init__(
            self,
            data,
            time_lengths=time_lengths,
            time_dim=time_dim
        )
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
            time_lengths: Optional[torch.LongTensor] = None,
            time_dim: Optional[int] = 0
    ) -> None:
        # Super call
        super(CharTimeTensor, self).__init__(
            self,
            data,
            time_lengths=time_lengths,
            time_dim=time_dim
        )
    # end __init__

# end CharTimeTensor

# endregion VARIANTS
