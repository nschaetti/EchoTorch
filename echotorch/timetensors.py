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
import torch

# Local
from .timetensor import TimeTensor


# Float time tensor
class FloatTimeTensor(TimeTensor):
    """Float time tensor.
    """

    # Constructor
    def __init__(
            self, data, time_dim=0, with_batch=False, **kwargs
    ):
        super(FloatTimeTensor, self).__init__(
            self, data, time_dim=time_dim, with_batch=with_batch, dtype=torch.float32, **kwargs
        )
    # end __init__

# end FloatTimeTensor


# Double time tensor
class DoubleTimeTensor(TimeTensor):
    """Double time tensor.
    """

    # Constructor
    def __init__(
            self, data, time_dim=0, with_batch=False, **kwargs
    ):
        super(DoubleTimeTensor, self).__init__(
            self, data, time_dim=time_dim, with_batch=with_batch, dtype=torch.float32, **kwargs
        )
    # end __init__

# end DoubleTimeTensor


# Half time tensor
class HalfTimeTensor(TimeTensor):
    """Half time tensor.
    """

    # Constructor
    def __init__(
            self, data, time_dim=0, with_batch=False, **kwargs
    ):
        super(HalfTimeTensor, self).__init__(
            self, data, time_dim=time_dim, with_batch=with_batch, dtype=torch.float16, **kwargs
        )
    # end __init__

# end HalfTimeTensor


# 16-bit floating point 2 time tensor
class BFloat16Tensor(TimeTensor):
    """16-bit floating point 2 time tensor.
    """

    # Constructor
    def __init__(
            self, data, time_dim=0, with_batch=False, **kwargs
    ):
        super(BFloat16Tensor, self).__init__(
            self, data, time_dim=time_dim, with_batch=with_batch, dtype=torch.float16, **kwargs
        )
    # end __init__

# end BFloat16Tensor


# 8-bit integer (unsigned) time tensor
class ByteTimeTensor(TimeTensor):
    """8-bit integer (unsigned) time tensor.
    """

    # Constructor
    def __init__(
            self, data, time_dim=0, with_batch=False, **kwargs
    ):
        super(ByteTimeTensor, self).__init__(
            self, data, time_dim=time_dim, with_batch=with_batch, dtype=torch.uint8, **kwargs
        )
    # end __init__

# end ByteTimeTensor


# 8-bit integer (signed) time tensor
class CharTimeTensor(TimeTensor):
    """8-bit integer (unsigned) time tensor.
    """

    # Constructor
    def __init__(
            self, data, time_dim=0, with_batch=False, **kwargs
    ):
        super(CharTimeTensor, self).__init__(
            self, data, time_dim=time_dim, with_batch=with_batch, dtype=torch.int8, **kwargs
        )
    # end __init__

# end CharTimeTensor
