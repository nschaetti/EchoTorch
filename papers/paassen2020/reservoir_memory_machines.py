# -*- coding: utf-8 -*-
#
# File : papers/paassen2020/reservoir_memory_machines.py
# Description : Reservoir Memory Machines (paassen2020)
# Date : 16th of July, 2020
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
import sys
import torch.utils.data
from torch.autograd import Variable
import torchvision.datasets
import echotorch.nn.reservoir
import matplotlib.pyplot as plt
import numpy as np

# Experiment paramters
cycle_weight = 1.0
jump_weight = 1.0
jump_size = 2

# Matrix factory
matrix_factory = echotorch.utils.matrix_generation.matrix_factory

# Reservoir matrix with cycle and jumps generator
w_generator = matrix_factory.get_generator(
    name='cycle_with_jumps',
    cycle_weight=cycle_weight,
    jumpy_weight=jump_weight,
    jump_size=jump_size
)

# Input matrix generation by aperiodic sequence and constant
win_generator = matrix_factory.get_generator(
    name='aperiodic_sequence',
    constant=1,
    start=0
)

print(np.array2string(win_generator.generate(size=(2, 10), dtype=torch.float64).numpy(), precision=2, separator=',', suppress_small=True))

