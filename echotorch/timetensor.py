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
from typing import Optional, Tuple, Union, List
import torch
import copy


# TimeTensor
class TimeTensor(object):
    """A  special tensor with a time and a batch dimension
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(
            self,
            data: torch.Tensor,
            time_dim: Optional[int] = 0,
            with_batch: Optional[bool] = False,
            with_channels: Optional[bool] = False,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            requires_grad: Optional[bool] = False
    ) -> None:
        """
        Constructor
        @param data:
        @param time_dim:
        @param with_batch:
        @param with_channels:
        @param dtype:
        @param device:
        @param requires_grad:
        """
        # Copy if already a timetensor
        # transform otherwise
        if type(data) is TimeTensor:
            # Copy
            tensor_data = copy.deepcopy(data.tensor)
        else:
            # Copy tensor
            tensor_data = copy.deepcopy(
                torch.as_tensor(data, dtype=dtype, device=device)
            )
        # end if

        # Set requires grad
        tensor_data.requires_grad = requires_grad

        # The tensor must have enough dimension
        # for the time dimension
        if tensor_data.ndim < time_dim + 1:
            # Error
            raise ValueError(
                "Time dimension does not exists in the data tensor "
                "(time dim at {}, {} dimension in tensor".format(time_dim, tensor_data.ndim)
            )
        # end if

        # If there is a batch dimension, time dimension cannot
        # be zero
        if with_batch and time_dim == 0:
            # Error
            raise ValueError("Time dimension cannot be the same as the batch dimension (batch is first)")
        # end if

        # Check dimension
        if with_batch and tensor_data.ndim <= 2:
            raise ValueError(
                "Time tensor with batch dimension must have "
                "at least two dimensions (here {}".format(tensor_data.ndim)
            )
        # end if

        # Properties
        self._tensor = tensor_data
        self._time_dim = time_dim
        self._with_batch = with_batch
    # end __init__

    # endregion CONSTRUCTORS

    # region PROPERTIES

    # Get tensor
    @property
    def tensor(self) -> torch.Tensor:
        """Get tensor
        """
        return self._tensor
    # end tensor

    # Time dimension (getter)
    @property
    def time_dim(self) -> int:
        """Time dimension (getter)
        """
        return self._time_dim
    # end time_dim

    # Time dimension (setter)
    @time_dim.setter
    def time_dim(self, value: int) -> None:
        """Time dimension (setter)
        """
        # Check time and batch dim not overlapping
        if self.with_batch and value == 0:
            raise ValueError("Time dimension cannot be the same as the batch dimension (batch is first)")
        # end if

        # Check time dim is valid
        if value >= self.tensor.ndim:
            # Error
            raise ValueError(
                "Time dimension does not exists in the data tensor "
                "(new time dim at {}, {} dimension in tensor".format(value, self._tensor.ndim)
            )
        # end if

        # Set new time dim
        self._time_dim = value
    # end time_dim

    # With batch (getter)
    @property
    def with_batch(self) -> bool:
        """With batch (getter)
        """
        return self._with_batch
    # end with_batch

    # Time length
    @property
    def tlen(self) -> int:
        """Time length
        """
        return self._tensor.size(self._time_dim)
    # end tlen

    # Batch size
    @property
    def batch_size(self) -> int:
        """Batch size
        """
        if self.with_batch:
            return self._tensor.size(0)
        else:
            return None
        # end if
    # end batch_size

    # Number of dimension
    @property
    def ndim(self):
        """Number of dimension
        """
        return self._tensor.ndim
    # end ndim

    # Number of channels
    @property
    def nchan(self):
        """Number of channels.
        """
        return self._time_dim - 1 if self._with_batch else self._time_dim
    # end nchan

    # Number of time-related dimension
    @property
    def tndim(self):
        """Number of time-related dimension.
        """
        return self._tensor.ndim - self._time_dim - 1
    # end tndim

    # Data type
    @property
    def dtype(self):
        """Data type.
        """
        return self._tensor.dtype
    # end dtype

    # Is on CUDA device
    @property
    def is_cuda(self):
        """Is True if the Tensor is stored on the GPU, False otherwise.
        """
        return self._tensor.is_cuda
    # end is_cuda

    # endregion PROPERTIES

    # region PUBLIC

    # Size
    def size(self):
        """Size
        """
        return self._tensor.size()
    # end size

    # Size of time-related dimension
    def tsize(self):
        """Size of time-related dimension
        """
        tensor_size = self._tensor.size()
        return tensor_size[self.time_dim+1:]
    # end tsize

    # Long
    def long(self):
        """Long.
        """
        self._tensor = self._tensor.long()
    # end long

    # Returns a new TimeTensor with data as the tensor data.
    def new_timetensor(
            data: torch.Tensor,
            time_dim: int = 0,
            with_batch: bool = False,
            with_channels: Optional[bool] = False,
            dtype: torch.dtype = None,
            device: torch.device = None,
            requires_grad: bool = False
    ) -> 'TimeTensor':
        """
        Returns a new TimeTensor with data as the tensor data.
        @param time_dim:
        @param with_batch:
        @param with_channels:
        @param dtype:
        @param device:
        @param requires_grad:
        @return:
        """
        return TimeTensor(
            data,
            time_dim=time_dim,
            with_batch=with_batch,
            with_channels=with_channels,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad
        )
    # end new_timetensor

    # Returns filled time tensor
    def new_full(
            *size: List[int],
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
        @param time_length:
        @param fill_value:
        @param n_channels:
        @param batch_size:
        @param dtype:
        @param device:
        @param requires_grad:
        @return:
        """
        # Size
        return TimeTensor.new_timetensor_with_func(
            size=size,
            func=torch.Tensor.new_full,
            time_length=time_length,
            n_channels=n_channels,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            fill_value=fill_value
        )
    # end new_full

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
            *size, time_length, n_channels, batch_size, with_batch=False, dtype=None, device=None,
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
            size: Tuple[int],
            time_length: int,
            n_channels: Optional[int] = None,
            batch_size: Optional[int] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            requires_grad: Optional[bool] = False
    ):
        """Returns a TimeTensor of size size filled with 0.
        """
        return TimeTensor.new_timetensor_with_func(
            size=size,
            func=torch.Tensor.new_zeros,
            time_length=time_length,
            n_channels=n_channels,
            batch_size=batch_size,
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

    # endregion PUBLIC

    # region OVERRIDE

    # To
    def to(self, *args, **kwargs) -> 'TimeTensor':
        """To.
        """
        self._tensor.to(*args, **kwargs)
        return self
    # end to

    # Get item
    def __getitem__(self, item):
        """Get item.
        """
        return self._tensor[item]
    # end __getitem__

    # Set item
    def __setitem__(self, key, value):
        """Set item."""
        self._tensor[key] = value
    # end __setitem__

    # Length
    def __len__(self):
        """Length
        """
        return self.tlen
    # end __len__

    # Get representation
    def __repr__(self):
        """Get representation
        """
        return "time dim:\n{}\n\nwith batch:\n{}\n\ndata:\n{}".format(self._time_dim, self._with_batch, self._tensor)
    # end __repr__

    # Torch functions
    def __torch_function__(self, func, types, args=(), kwargs=None):
        """Torch functions
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

        # Get the tensor in the arguments
        args = [convert(a) for a in args]

        # Execute function
        ret = func(*args, **kwargs)

        # Return a new time tensor
        return TimeTensor(ret, time_dim=self._time_dim, with_batch=self._with_batch)
    # end __torch_function__

    # endregion OVERRIDE

    # region STATIC

    # endregion STATIC

# end TimeTensor
