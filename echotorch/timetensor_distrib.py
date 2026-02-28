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


from typing import Optional, Sequence, Union

from . import base_ops


# Returns a timetensor filled with random numbers from a uniform distribution on the interval [0, 1)[0,1)
def rand(
        size: Union[int, Sequence[int]],
        time_length: int,
        time_first: Optional[bool] = True,
        **kwargs
):
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
    if isinstance(size, int):
        size = (size,)
    return base_ops.rand(*tuple(size), length=time_length, **kwargs)
# end rand

