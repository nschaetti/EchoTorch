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
        time_first: Optional[bool] = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False
) -> TimeTensor:
    """
    Returns a timetensor filled with random numbers from a uniform distribution on the interval [0, 1)[0,1)
    @param size:
    @param time_length:
    @param time_first:
    @param dtype:
    @param device:
    @param requires_grad:
    @return:
    """
    return TimeTensor.new_timetensor_with_func(
        size,
        func=torch.rand,
        time_length=time_length,
        time_first=time_first,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad
    )
# end rand



