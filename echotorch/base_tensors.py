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

    # Get tensor
    @property
    def tensor(self) -> torch.Tensor:
        r"""Get the wrapped tensor.

        :return: The wrapped tensor.
        :rtype: :class:`torch.Tensor`
        """
        return self._tensor
    # end tensor

    # endregion PROPERTIES

    # region CAST

    # To float64 basetensor
    def double(self) -> 'BaseTensor':
        r"""To float64 :class:`BaseTensor` (no copy).

        :return: The :class:`BaseTensor` with data casted to float64.
        :rtype: :class:`BaseTensor`
        """
        self._tensor = self._tensor.double()
        return self
    # end double

    # To float32 basetensor
    def float(self) -> 'BaseTensor':
        r"""To float32 :class:`BaseTensor` (no copy).

        :return: The :class:`BaseTensor` with data casted to float32.
        :rtype: :class:`BaseTensor`
        """
        self._tensor = self._tensor.float()
        return self
    # end float

    # To float16 basetensor
    def half(self) -> 'BaseTensor':
        r"""To float16 :class:`BaseTensor` (no copy)

        :return: The :class:`BaseTensor` with data casted to float16.
        :rtype: :class:`BaseTensor`
        """
        self._tensor = self._tensor.half()
        return self
    # end half

    # To bfloat16 basetensor
    def bfloat16(self) -> 'BaseTensor':
        r"""To brain float16 :class:`BaseTensor` (no copy)

        :return: The :class:`BaseTensor` with data casted to bfloat16.
        :rtype: :class:`BaseTensor`
        """
        self._tensor = self._tensor.bfloat16()
    # end bfloat16

    # To boolean basetensor
    def bool(self) -> 'BaseTensor':
        r"""To boolean :class:`BaseTensor` (no copy).

        :return: The :class:`BaseTensor` with data casted to boolean.
        :rtype: :class:`BaseTensor`
        """
        self._tensor = self._tensor.bool()
        return self
    # end bool

    # To byte basetensor
    def byte(self) -> 'BaseTensor':
        r"""To byte :class:`BaseTensor` (no copy).

        :return: The :class:`BaseTensor` with data casted to bytes.
        :rtype: :class:`BaseTensor`
        """
        self._tensor = self._tensor.byte()
        return self
    # end byte

    # To char basetensor
    def char(self) -> 'BaseTensor':
        r"""To char :class:`BaseTensor` (no copy)

        :return: The :class:`BaseTensor` with data casted to char
        :rtype: :class:`BaseTensor`
        """
        self._tensor = self._tensor.char()
        return self
    # end char

    # To short basetensor
    def short(self) -> 'BaseTensor':
        r"""To short (int16) :class:`BaseTensor` (no copy)

        :return: The :class:`BaseTensor` with data casted to char
        :rtype: :class:`BaseTensor`
        """
        self._tensor = self._tensor.char()
        return self
    # end char

    # To int timetensor
    def int(self) -> 'BaseTensor':
        r"""To int :class:`BaseTensor` (no copy)

        :return: The :class:`BaseTensor` with data casted to int.
        :rtype: :class:`BaseTensor`
        """
        self._tensor = self._tensor.int()
        return self
    # end int

    # Long
    def long(self) -> 'BaseTensor':
        r"""To long :class:`BaseTensor` (no copy)

        :return: The :class:`BaseTensor` with data casted to long
        :rtype: :class:`BaseTensor`
        """
        self._tensor = self._tensor.long()
        return self
    # end long

    # To
    def to(self, *args, **kwargs) -> 'BaseTensor':
        r"""Performs BaseTensor dtype and/or device concersion. A ``torch.dtype`` and ``torch.device`` are inferred
        from the arguments of ``self.to(*args, **kwargs)

        .. note::
            From PyTorch documentation: if the ``self`` BaseTensor already has the correct ``torch.dtype`` and
            ``torch.device``, then ``self`` is returned. Otherwise, the returned basetensor is a copy of ``self``
            with the desired ``torch.dtype`` and ``torch.device``.
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

    # To CUDA device
    def cuda(
            self,
            **kwargs
    ) -> 'BaseTensor':
        r"""To CUDA device.

        :return: BaseTensor transfered to GPU device.
        :rtype: :class:`BaseTensor`
        """
        self._tensor = self._tensor.cuda(**kwargs)
        return self
    # end cuda

    # To CPU device
    def cpu(
            self,
            **kwargs
    ) -> 'BaseTensor':
        r"""To CPU device.

        :return: BaseTensor transferred to CPU device.
        :rtype: :class:`BaseTensor`
        """
        self._tensor = self._tensor.cpu(**kwargs)
        return self
    # end cpu

    # region TORCH_FUNCTION

    # Transpose
    def t(self) -> 'BaseTensor':
        r"""Expects the basetensor to be <= 2-D tensor and transposes dimensions 0 and 1.

        0-D and 1-D tensors are returned as is. When input is a 2-D tensor this is equivalent to ``transpose(input, 0, 1)``.
        """
        self._tensor = self._tensor.t()
        return self
    # end t

    # Torch functions
    def __torch_function__(
            self,
            func,
            types,
            args=(),
            kwargs=None
    ):
        r"""Torch functions implementations.
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

    # endregion TORCH_FUNCTION

    # region OVERRIDE

    # Override attribute getter
    def __getattr__(self, item):
        r"""Override attribute getter and redirect unknown attributes to wrapper tensor.
        """
        if hasattr(self._tensor, item):
            return getattr(self._tensor, item)
        else:
            raise AttributeError(
                "AttributeError: Neither '{}' object nor its wrapped "
                "tensor has no attribute '{}'".format(self.__class__.__name__, item)
            )
        # end if
    # end __getattr__

    # Get item
    def __getitem__(self, item) -> 'BaseTensor':
        r"""Get data in the :class:`BaseTensor`.
        """
        return BaseTensor(self._tensor[item])
    # end __getitem__

    # Set item
    def __setitem__(self, key, value) -> None:
        r"""Set data in the :class:`BaseTensor`.
        """
        self._tensor[key] = value
    # end __setitem__

    # Get representation
    def __repr__(self) -> str:
        r"""Get the :class:`BaseTensor` string representation
        """
        return "basetensor({})".format(self._tensor)
    # end __repr__

    # Are two time-tensors equivalent
    def __eq__(
            self,
            other: 'BaseTensor'
    ) -> bool:
        r"""Are two :class:`BaseTensor` equivalent?

        :param other: The other :class:`BaseTensor`.
        :return: True of False if the two :class:`BaseTensor` are equivalent.
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
        self._tensor -= other
        return self
    # end __isub__

    # Scalar addition
    def __add__(self, other):
        r"""Scalar addition with time tensors.

        :param other: Scalar to add
        :type other: Scalar
        """
        if isinstance(other, BaseTensor):
            self._tensor = self._tensor + other.tensor
        else:
            self._tensor = self._tensor + other
        # end if

        return self
    # end __add__

    # Scalar addition (right)
    def __radd__(self, other):
        r"""Scalar addition with time tensors (right)

        :param other: Scalar to add
        :type other: Scalar
        """
        if isinstance(other, BaseTensor):
            self._tensor = self._tensor + other.tensor
        else:
            self._tensor = self._tensor + other
        # end if

        return self
    # end __radd__

    # Scalar subtraction
    def __sub__(self, other):
        r"""Scalar subtraction with time tensors.

        :param other: Scalar to subtract.
        :type other: scalar
        """
        if isinstance(other, BaseTensor):
            self._tensor = self._tensor - other.tensor
        else:
            self._tensor = self._tensor - other
        # end if

        return self
    # end __sub__

    # Scalar subtraction (right)
    def __rsub__(self, other):
        r"""Scalar subtraction with time tensors (right).

        :param other: Scalar to subtract.
        :type other: scalar.
        """
        if isinstance(other, BaseTensor):
            self._tensor = self._tensor - other.tensor
        else:
            self._tensor = self._tensor - other
        # end if

        return self
    # end __rsub__

    # Scalar multiplication
    def __mul__(self, other):
        r"""Scalar multiplication with time tensors

        :param other: Scalar multiplier
        :type other: Scalar
        """
        if isinstance(other, BaseTensor):
            self._tensor = self._tensor * other.tensor
        else:
            self._tensor = self._tensor * other
        # end if
        return self
    # end __mul__

    # Scalar multiplication (right)
    def __rmul__(self, other):
        r"""Scalar multiplication with time tensors

        :param other: Scalar multiplier
        :param type: Scalar
        """
        if isinstance(other, BaseTensor):
            self._tensor = self._tensor * other.tensor
        else:
            self._tensor = self._tensor * other
        # end if

        return self
    # end __rmul__

    def __truediv__(self, other):
        r"""Scalar division with time tensors.

        :param other: Scalar divisor.
        :param type: Scalar.
        """
        if isinstance(other, BaseTensor):
            self._tensor = self._tensor / other.tensor
        else:
            self._tensor = self._tensor / other
        # end if

        return self
    # end __truediv__

    # endregion OVERRIDE

    # region STATIC

    # Returns a new BaseTensor with data as the tensor data.
    @classmethod
    def new_basetensor(
            cls,
            data: Union[torch.Tensor, 'BaseTensor']
    ) -> 'BaseTensor':
        r"""Returns a new :class:`BaseTensor` with data as the tensor data.

        :param data: data as a torch tensor or another :class:`BaseTensor`.
        :return: a new :class:`BaseTensor` with *data*.
        :rtype: :class:`BaseTensor`
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
        r"""Returns a new base tensor with a specific function to generate the data.

        :param func: a callable object used for creation.
        :param size: size of the :class:`BaseTensor` to be created.
        :return: a new :class:`BaseTensor` created with ``func`` of size ``size``.
        :rtype: :class:`BaseTensor`
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


# Double time tensor
class DoubleBaseTensor(BaseTensor):
    r"""Double :class:``BaseTensor`.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'BaseTensor']
    ) -> None:
        r"""Double :class:``BaseTensor` constructor

        Args:
            data: The data in a torch tensor to transform to :class:``BaseTensor`.
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


# Float base tensor
class FloatBaseTensor(BaseTensor):
    r"""Float :class:``BaseTensor`.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'BaseTensor'],
    ) -> None:
        r"""Float :class:``BaseTensor` constructor

        :param data: The data in a torch tensor to transform to :class:``BaseTensor`.
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


# Half base tensor
class HalfBaseTensor(BaseTensor):
    r"""Half :class:``BaseTensor`.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'BaseTensor']
    ) -> None:
        r"""Half :class:``BaseTensor` constructor

        :param data: The data in a torch tensor to transform to :class:``BaseTensor`.
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
    r"""16-bit floating point 2 :class:``BaseTensor`.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'BaseTensor']
    ) -> None:
        r"""16-bit :class:``BaseTensor` constructor

        :param data: The data in a torch tensor to transform to :class:``BaseTensor`.
        """
        # Super call
        super(BFloat16Tensor, self).__init__(
            self,
            data
        )

        # Cast
        self.bfloat16()
    # end __init__

# end BFloat16Tensor


# Boolean basetensor
class BoolBaseTensor(BaseTensor):
    r"""To boolean :class:``BaseTensor`.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'BaseTensor']
    ) -> None:
        r"""Boolean :class:``BaseTensor`.
        """
        # Super call
        super(BoolBaseTensor, self).__init__(
            self,
            data
        )

        # Cast
        self.bool()
    # end __init__

# end BoolBaseTensor


# 8-bit integer (unsigned) base tensor
class ByteBaseTensor(BaseTensor):
    r"""8-bit integer (unsigned) :class:``BaseTensor`.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'BaseTensor']
    ) -> None:
        r"""8-bit integer (unsigned) :class:``BaseTensor` constructor

        :param data: The data in a torch tensor to transform to :class:``BaseTensor`.
        """
        # Super call
        super(ByteBaseTensor, self).__init__(
            self,
            data
        )

        # Cast
        self.byte()
    # end __init__

# end ByteBaseTensor


# 8-bit integer (signed) base tensor
class CharBaseTensor(BaseTensor):
    r"""8-bit integer :class:``BaseTensor`.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'BaseTensor']
    ) -> None:
        r"""8-bit integer :class:``BaseTensor`.
        """
        # Super call
        super(CharBaseTensor, self).__init__(
            self,
            data
        )

        # Case
        self.char()
    # end __init__

# end CharTimeTensor


# 16-bit integer (signed) base tensor.
class ShortBaseTensor(BaseTensor):
    r"""16-bit integer (signed) :class:``BaseTensor`.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'BaseTensor']
    ) -> None:
        r"""8-bit integer :class:``BaseTensor`.
        """
        # Super call
        super(ShortBaseTensor, self).__init__(
            self,
            data
        )

        # Cast
        self.short()
    # end __init__

# end ShortBaseTensor


# 32-bit integer (signed) base tensor.
class IntBaseTensor(BaseTensor):
    r"""32-bit integer (signed) :class:``BaseTensor`.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'BaseTensor']
    ) -> None:
        r"""32-bit integer :class:``BaseTensor`.
        """
        # Super call
        super(IntBaseTensor, self).__init__(
            self,
            data
        )

        # Cast
        self.int()
    # end __init__

# end IntBaseTensor


# 64-bit integer (signed) base tensor.
class LongBaseTensor(BaseTensor):
    r"""64-bit integer (signed) :class:``BaseTensor`.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'BaseTensor']
    ) -> None:
        r"""64-bit integer :class:``BaseTensor`.
        """
        # Super call
        super(LongBaseTensor, self).__init__(
            self,
            data
        )

        # Cast
        self.long()
    # end __init__

# end LongBaseTensor


# endregion VARIANTS


