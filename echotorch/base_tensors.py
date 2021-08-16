# -*- coding: utf-8 -*-
#
# File : echotorch/base_tensor.py
# Description : An abstract base class for EchoTorch tensors
# Date : 13th of August, 2021
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
from typing import Tuple, Union, Callable
import torch
import numpy as np


# region BASETENSOR

class BaseTensor(object):
    r"""An abstract base class for EchoTorch tensors
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, data: Union[torch.Tensor, 'TimeTensor']) -> None:
        r"""BaseTensor constructor

        :param data: The data in a torch tensor to transform to timetensor.
        :type data: ``torch.Tensor`` or ``DataTensor``
        """
        # Copy if already a timetensor
        # transform otherwise
        if isinstance(data, BaseTensor):
            tensor_data = data.tensor
        else:
            tensor_data = data
        # end if

        # Set tensor and time index
        self._tensor = tensor_data
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

    # Number of dimension
    @property
    def ndim(self) -> int:
        """
        Number of dimension
        """
        return self._tensor.ndim
    # end ndim

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

    # Long
    def long(self) -> 'BaseTensor':
        r"""To long BaseTensor (no copy)

        Returns: The BaseTensor with data casted to long

        """
        self._tensor = self._tensor.long()
        return self
    # end long

    # Float
    def float(self) -> 'BaseTensor':
        r"""To float32 BaseTensor (no copy)

        Returns: The BaseTensor with data coasted to float32

        """
        self._tensor = self._tensor.float()
        return self
    # end float

    # Complex
    def complex(self) -> 'BaseTensor':
        r"""To complex BaseTensor (no copy)

        Returns: The BaseTensor with data casted to complex

        """
        self._tensor = self._tensor.complex()
        return self
    # end complex

    # To float16 timetensor
    def half(self) -> 'BaseTensor':
        r"""To float16 BaseTensor (no copy)

        Returns: The BaseTensor with data casted to float16

        """
        self._tensor = self._tensor.half()
        return self
    # end half

    # To
    def to(self, *args, **kwargs) -> 'BaseTensor':
        r"""Performs BaseTensor dtype and/or device concersion. A ``torch.dtype`` and ``torch.device`` are inferred
        from the arguments of ``self.to(*args, **kwargs)

        .. note::
            From PyTorch documentation: if the ``self`` BaseTensor already has the correct ``torch.dtype`` and
            ``torch.device``, then ``self`` is returned. Otherwise, the returned basetensor is a copy of ``self``
            with the desired ``torch.dtype`` and ``torch.device``.

        Args:
            *args:
            **kwargs:

        """
        # New tensor
        ntensor = self._tensor.to(*args, **kwargs)

        # Same tensor?
        if self._tensor == ntensor:
            return self
        else:
            return BaseTensor(ntensor)
        # end if
    # end to

    # endregion PUBLIC

    # region TORCH_FUNCTION

    # endregion TORCH_FUNCTION

    # region OVERRIDE

    # Transpose
    def t(self) -> torch.Tensor:
        r"""Expects the basetensor to be <= 2-D tensor and transposes dimensions 0 and 1.

        0-D and 1-D tensors are returned as is. When input is a 2-D tensor this is equivalent to ``transpose(input, 0, 1)``.
        """
        return self._tensor.t()
    # end t

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
    ) -> 'BaseTensor':
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
    ) -> 'BaseTensor':
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
    ) -> 'BaseTensor':
        """
        Transfer to device
        """
        self._tensor.to(*args, **kwargs)
        return self
    # end to

    # Get item
    def __getitem__(self, item) -> 'BaseTensor':
        """
        Get data in the tensor
        """
        return BaseTensor(self._tensor[item])
    # end __getitem__

    # Set item
    def __setitem__(self, key, value) -> None:
        """
        Set data in the tensor
        """
        self._tensor[key] = value
    # end __setitem__

    # Get representation
    def __repr__(self) -> str:
        """
        Get a string representation
        """
        return "basetensor({})".format(self._tensor)
    # end __repr__

    # Are two time-tensors equivalent
    def __eq__(
            self,
            other: 'BaseTensor'
    ) -> bool:
        """
        Are two time-tensors equivalent?
        @param other: The other time-tensor
        @return: True of False if the two time-tensors are equivalent
        """
        return self.tensor.ndim == other.tensor.ndim and self.tensor.size() == other.tensor.size() and \
               torch.all(self.tensor == other.tensor)
    # end __eq__

    # Object addition
    def __iadd__(self, other):
        r"""Object addition with time tensors.

        :param other: object to add
        :type other: ``TimeTensor`` or ``torch.Tensor``
        """
        self._tensor += other
        return self
    # end __iadd__

    # Object substraction
    def __isub__(self, other):
        r"""Object subtraction with time tensors.

        :param other: object to add
        :type other: ``TimeTensor`` or ``torch.Tensor``
        """
        self._tensor += other
        return self
    # end __isub__

    # Scalar addition
    def __add__(self, other):
        r"""Scalar addition with time tensors.

        :param other: Scalar to add
        :type other: Scalar
        """
        self._tensor += other
        return self
    # end __add__

    # Scalar addition (right)
    def __radd__(self, other):
        r"""Scalar addition with time tensors (right)

        :param other: Scalar to add
        :type other: Scalar
        """
        self._tensor += other
        return self
    # end __radd__

    # Scalar subtraction
    def __sub__(self, other):
        r"""Scalar subtraction with time tensors.

        :param other: Scalar to subtract.
        :type other: scalar
        """
        self._tensor -= other
        return self
    # end __sub__

    # Scalar subtraction (right)
    def __rsub__(self, other):
        r"""Scalar subtraction with time tensors (right).

        :param other: Scalar to subtract.
        :type other: scalar.
        """
        self._tensor -= other
        return self
    # end __rsub__

    # Scalar multiplication
    def __mul__(self, other):
        r"""Scalar multiplication with time tensors

        :param other: Scalar multiplier
        :type other: Scalar
        """
        self._tensor *= other
        return self
    # end __mul__

    # Scalar multiplication (right)
    def __rmul__(self, other):
        r"""Scalar multiplication with time tensors

        :param other: Scalar multiplier
        :param type: Scalar
        """
        self._tensor *= other
        return self
    # end __rmul__

    def __truediv__(self, other):
        r"""Scalar division with time tensors.

        :param other: Scalar divisor.
        :param type: Scalar.
        """
        self._tensor /= other
        return self
    # end __truediv__

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
            if type(args) is BaseTensor:
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

        # Return a new base tensor
        return BaseTensor(ret)
    # end __torch_function__

    # endregion OVERRIDE

    # region STATIC

    # Returns a new BaseTensor with data as the tensor data.
    @classmethod
    def new_basetensor(
            cls,
            data: Union[torch.Tensor, 'BaseTensor']
    ) -> 'BaseTensor':
        """
        Returns a new TimeTensor with data as the tensor data.
        @param data:
        @return:
        """
        return BaseTensor(
            data
        )
    # end new_basetensor

    # Returns new base tensor with a specific function
    @classmethod
    def new_basetensor_with_func(
            cls,
            size: Tuple[int],
            func: Callable,
            **kwargs
    ) -> 'BaseTensor':
        """
        Returns a new base tensor with a specific function to generate the data.
        @param func:
        @param size:
        """
        # Create BaseTensor
        return BaseTensor(
            data=func(size, **kwargs)
        )
    # end new_basetensor_with_func

    # endregion STATIC

# end BaseTensor

# endregion BASETENSOR


# region VARIANTS

# Float base tensor
class FloatBaseTensor(BaseTensor):
    r"""Float base tensor.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'BaseTensor'],
    ) -> None:
        r"""Float BaseTensor constructor

        Args:
            data: The data in a torch tensor to transform to basetensor.
        """
        # Super call
        super(FloatBaseTensor, self).__init__(
            self,
            data
        )

        # Transform type
        self.float()
    # end __init__

# end FloatBaseTensor


# Double time tensor
class DoubleBaseTensor(BaseTensor):
    r"""Double time tensor.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'BaseTensor']
    ) -> None:
        r"""Double BaseTensor constructor

        Args:
            data: The data in a torch tensor to transform to basetensor.
        """
        # Super call
        super(DoubleBaseTensor, self).__init__(
            self,
            data
        )

        # Cast data
        self.double()
    # end __init__

# end DoubleBaseTensor


# Half base tensor
class HalfBaseTensor(BaseTensor):
    r"""Half base tensor.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'BaseTensor']
    ) -> None:
        r"""Half BaseTensor constructor

        Args:
            data: The data in a torch tensor to transform to basetensor.
        """
        # Super call
        super(HalfBaseTensor, self).__init__(
            self,
            data
        )

        # Cast data
        self.half()
    # end __init__

# end HalfBaseTensor


# 16-bit floating point 2 base tensor
class BFloat16Tensor(BaseTensor):
    r"""16-bit floating point 2 base tensor.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'BaseTensor']
    ) -> None:
        r"""16-bit BaseTensor constructor

        Args:
            data: The data in a torch tensor to transform to basetensor.
        """
        # Super call
        super(BFloat16Tensor, self).__init__(
            self,
            data
        )
    # end __init__

# end BFloat16Tensor


# 8-bit integer (unsigned) base tensor
class ByteBaseTensor(BaseTensor):
    r"""8-bit integer (unsigned) base tensor.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'BaseTensor']
    ) -> None:
        r"""8-bit integer (unsigned) BaseTensor constructor

        Args:
            data: The data in a torch tensor to transform to basetensor.
        """
        # Super call
        super(ByteBaseTensor, self).__init__(
            self,
            data
        )
    # end __init__

# end ByteBaseTensor


# 8-bit integer (signed) base tensor
class CharBaseTensor(BaseTensor):
    r"""8-bit integer base tensor.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'BaseTensor']
    ) -> None:
        r"""8-bit integer base tensor.
        """
        # Super call
        super(CharBaseTensor, self).__init__(
            self,
            data
        )
    # end __init__

# end CharTimeTensor

# endregion VARIANTS


