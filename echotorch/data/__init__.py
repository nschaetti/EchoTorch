# -*- coding: utf-8 -*-
#
# File : echotorch/data/__init__.py
# Description : Dataset subpackages init file
# Date : 3th of March, 2021
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

# Import functions
from .random_processes import random_walk, moving_average, autoregressive_process, autoregressive_moving_average
from .chaotic import henon

# Alias
rw = random_walk
ma = moving_average
ar = autoregressive_process
arma = autoregressive_moving_average

__all__ = [
   # Chaotic
   'henon',
   # Random process
   'random_walk', 'moving_average', 'autoregressive_process', 'autoregressive_moving_average',
   'rw', 'ma', 'ar', 'arma'
]
