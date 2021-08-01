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
import copy


# Error
ERROR_TENSOR_TO_SMALL = "Time dimension does not exists in the data tensor " \
                        "(time dim at {}, {} dimension in tensor). The minimum tensor size " \
                        "is {}"


# TimeTensor
class TimeTensor(object):
    """
    A  special tensor with a time and a batch dimension
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(
            self,
            data: torch.Tensor,
            time_dim: Optional[int] = 0,
            time_first: Optional[bool] = True,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            requires_grad: Optional[bool] = True,
            copy_data: Optional[bool] = True
    ) -> None:
        """
        Constructor
        @param data: The data in a torch tensor to transform to timetensor.
        @param time_dim: The position of the time dimension.
        @param time_first: Is the time dimension place before the channel or after?
        @param dtype: Data tensor type
        @param device: Destination device
        @param requires_grad: Requires gradient computation?
        """
        # Copy if already a timetensor
        # transform otherwise
        if type(data) is TimeTensor:
            if copy_data:
                # Copy
                tensor_data = copy.deepcopy(data.tensor)
            else:
                tensor_data = data.tensor
            # end if
        else:
            if copy_data:
                # Copy
                if isinstance(data, torch.Tensor) and not data.is_leaf:
                    tensor_data = torch.as_tensor(data, dtype=dtype, device=device)
                else:
                    # Copy tensor
                    tensor_data = copy.deepcopy(
                        torch.as_tensor(data, dtype=dtype, device=device)
                    )
                # end if
            else:
                tensor_data = torch.as_tensor(data, dtype=dtype, device=device)
            # end if
        # end if

        # Set requires grad
        if tensor_data.is_leaf:
            tensor_data.requires_grad = requires_grad
        # end if

        # The tensor must have enough dimension
        # for the time dimension
        if tensor_data.ndim < time_dim + 1:
            # Error
            raise ValueError(ERROR_TENSOR_TO_SMALL.format(time_dim, tensor_data.ndim, time_dim + 1))
        # end if

        # Properties
        self._tensor = tensor_data
        self._time_dim = time_dim
        self._time_first = time_first
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
        return self._tensor.size(self._time_dim)
    # end tlen

    # Number of dimension
    @property
    def ndim(self) -> int:
        """
        Number of dimension
        """
        return self._tensor.ndim
    # end ndim

    # Number of time-related dimension
    @property
    def tndim(self) -> int:
        """
        Number of time-related dimension.
        """
        return self._tensor.ndim - self._time_dim - 1
    # end tndim

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

    # Size of time-related dimension
    def tsize(self) -> torch.Size:
        """
        Size of time-related dimensions
        """
        if self._time_dim != self._tensor.ndim - 1:
            tensor_size = self._tensor.size()
            return tensor_size[self.time_dim+1:]
        else:
            return torch.Size([])
        # end if
    # end tsize

    # Long
    def long(self) -> 'TimeTensor':
        """
        Long
        """
        self._tensor = self._tensor.long()
        return self
    # end long

    # Float
    def float(self) -> 'TimeTensor':
        """
        Cast tensor to float
        """
        self._tensor = self._tensor.float()
        return self
    # end float

    # Returns a new TimeTensor with data as the tensor data.
    def new_timetensor(
            data: torch.Tensor,
            time_dim: Optional[int] = 0,
            time_first: Optional[bool] = True,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            requires_grad: Optional[bool] = False
    ) -> 'TimeTensor':
        """
        Returns a new TimeTensor with data as the tensor data.
        @param data: The data in a torch tensor to transform to timetensor.
        @param time_dim: The position of the time dimension.
        @param time_first: Is the time dimension place before the channel or after?
        @param dtype: Data tensor type
        @param device: Destination device
        @param requires_grad: Requires gradient computation?
        @return: A new timetensor
        """
        return TimeTensor(
            data,
            time_dim=time_dim,
            time_first=time_first,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad
        )
    # end new_timetensor

    # Returns filled time tensor
    def new_full(
            size: List[int],
            fill_value: Union[int, float],
            time_length: int = 0,
            time_first: Optional[bool] = True,
            dtype: torch.dtype = None,
            device: torch.device = None,
            requires_grad: bool = False
    ) -> 'TimeTensor':
        """
        Returns a TimeTensor of size size and time length time_length filled with fill_value. By default,
        the returned Tensor has the same torch.dtype and torch.device as this tensor.
        @param time_first:
        @param size: Size if the timeseries
        @param time_length: Time-length of the timeseries
        @param fill_value: Value to fill the tensor
        @param dtype: Tensor data type
        @param device: Destination device
        @param requires_grad: Requires gradient computation
        @return: The new timetensor
        """
        # Size
        return TimeTensor.new_timetensor_with_func(
            size=size,
            func=torch.full,
            time_length=time_length,
            time_first=time_first,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            fill_value=fill_value
        )
    # end new_full

    # Returns empty tensor
    def new_empty(
            size: Tuple[int],
            time_length: int,
            time_first: Optional[bool] = True,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            requires_grad: Optional[bool] = False
    ) -> 'TimeTensor':
        """
        Returns a TimeTensor of size size and time length time_length filled with uninitialized data. By default,
        the returned TimeTensor has the same torch.dtype and torch.device as this tensor.
        @param size:
        @param time_length:
        @param time_first:
        @param dtype:
        @param device:
        @param requires_grad:
        """
        return TimeTensor.new_timetensor_with_func(
            size,
            func=torch.empty,
            time_length=time_length,
            time_first=time_first,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    # end new_empty

    # Returns time tensor filled with ones
    def new_ones(
            size,
            time_length,
            time_first: Optional[bool] = True,
            dtype=None,
            device=None,
            requires_grad=False
    ) -> 'TimeTensor':
        """
        Returns a TimeTensor of size size filled with 1. By default, the returned TimeTensor has the same
        torch.dtype and torch.device as this tensor.
        @param size:
        @param time_length:
        @param time_first:
        @param dtype:
        @param device:
        @param requires_grad:
        """
        return TimeTensor.new_timetensor_with_func(
            size,
            func=torch.ones,
            time_length=time_length,
            time_first=time_first,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
    # end new_ones

    # Returns time tensor filled with zeros
    def new_zeros(
            size: Tuple[int],
            time_length: int,
            time_first: Optional[bool] = True,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            requires_grad: Optional[bool] = False
    ):
        """
        Returns a TimeTensor of size size filled with 0.
        @param size:
        @param time_length:
        @param time_first:
        @param dtype:
        @param device:
        @param requires_grad:
        """
        return TimeTensor.new_timetensor_with_func(
            size=size,
            func=torch.zeros,
            time_length=time_length,
            time_first=time_first,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
    # end new_zeros

    # Returns new time tensor with a specific function
    def new_timetensor_with_func(
            size: Tuple[int],
            func: Callable,
            time_length: int,
            time_first: Optional[bool] = True,
            **kwargs
    ) -> 'TimeTensor':
        """
        Returns a new time tensor with a specific function to generate the data.
        @param size:
        @param time_length:
        @param time_first:
        @param dtype:
        @param device:
        @param requires_grad:
        """
        # Size
        tt_size = [time_length] + list(size) if time_first else list(size) + [time_length]

        # Create TimeTensor
        return TimeTensor(
            func(tuple(tt_size), **kwargs),
            time_dim=0 if time_first else len(tt_size) - 1,
            time_first=time_first,
            **kwargs
        )
    # end new_timetensor_with_func

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
            time_dim=self._time_dim,
            time_first=self._time_first,
            dtype=self._tensor.dtype,
            device=self._tensor.device,
            requires_grad=self._tensor.requires_grad,
            copy_data=False
        )
    # end indexing_timetensor

    # endregion PUBLIC

    # region TORCH_FUNCTION

    # After unsqueeze
    def after_unsqueeze(
            self,
            func_output: Any,
            dim
    ) -> 'TimeTensor':
        """
        After unsqueeze
        @param func_output:
        @param dim:
        @return:
        """
        return TimeTensor(
            func_output,
            time_dim=self._time_dim,
            time_first=self._time_first,
            device=self.device,
        )
    # end after_unsqueeze

    # endregion TORCH_FUNCTION

    # region OVERRIDE

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
        return "time dim:\n{}\n\ndata:\n{}".format(self._time_dim, self._tensor)
    # end __repr__

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
        elif isinstance(ret, TimeTensor) or isinstance(ret, torch.Tensor):
            # Return a new time tensor
            return TimeTensor(
                ret,
                time_dim=self._time_dim,
                time_first=self._time_first,
                device=self.device,
            )
        else:
            return ret
        # end if
    # end __torch_function__

    # endregion OVERRIDE

    # region STATIC

    # endregion STATIC

# end TimeTensor
