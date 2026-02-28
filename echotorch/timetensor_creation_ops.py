# -*- coding: utf-8 -*-
#
# File : echotorch/timetensor_creation_ops.py
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


def timetensor(*args, **kwargs):
    return base_ops.timetensor(*args, **kwargs)


def as_timetensor(*args, **kwargs):
    return base_ops.as_timetensor(*args, **kwargs)


def from_numpy(*args, **kwargs):
    return base_ops.from_numpy(*args, **kwargs)


def full(
        size: Union[int, Sequence[int]],
        fill_value,
        time_length: int,
        **kwargs
):
    if isinstance(size, int):
        size = (size,)
    return base_ops.full(*tuple(size), fill_value=fill_value, length=time_length, **kwargs)


def empty(
        size: Union[int, Sequence[int]],
        time_length: int,
        **kwargs
):
    if isinstance(size, int):
        size = (size,)
    return base_ops.empty(*tuple(size), length=time_length, **kwargs)


def ones(
        size: Union[int, Sequence[int]],
        time_length: int,
        **kwargs
):
    if isinstance(size, int):
        size = (size,)
    return base_ops.ones(*tuple(size), length=time_length, **kwargs)


def zeros(
        size: Union[int, Sequence[int]],
        time_length: int,
        **kwargs
):
    if isinstance(size, int):
        size = (size,)
    return base_ops.zeros(*tuple(size), length=time_length, **kwargs)
