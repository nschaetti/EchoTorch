# -*- coding: utf-8 -*-
#
# File : echotorch/timetensor_creation.py
# Description : TimeTensor creation helper functions
# Date : 27th of Jully, 2021
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
from typing import Tuple, Optional
import torch

# Import local
from .timetensor import TimeTensor


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



