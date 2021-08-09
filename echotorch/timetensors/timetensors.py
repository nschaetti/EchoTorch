# -*- coding: utf-8 -*-
#
# File : echotorch/timetensors.py
# Description : Special tensors
# Date : 5th of May, 2021
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
from typing import Union, Optional
import torch

# Local
from .timetensor import TimeTensor


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
