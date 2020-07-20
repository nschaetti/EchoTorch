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
import torch.utils.data
import echotorch.nn.reservoir
import echotorch.datasets as etda
import matplotlib.pyplot as plt
import numpy as np

# Experiment parameters
reservoir_size = 128
input_scaling = 0.1
bias_scaling = 0.0
leak_rate = 1.0
memory_size = 16
ridge_param = 0.00001
input_normalization = True
washout = 0
cycle_weight = 0.9
jump_weight = 0.3
jump_size = 2

# Task parameters
n_folds = 20
seq_per_fold = 20
min_len = 50
max_len = 200
n_switches = 3

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

# Bias matrix generation with zero
wbias_generator = matrix_factory.get_generator(
    connectivity=1.0,
    scale=bias_scaling,
    apply_spectral_radius=False
)

# Init. random number generators
echotorch.utils.manual_seed(1)

# Latch task dataset
latch_task_dataset = etda.LatchTaskDataset(
    n_samples=n_folds * seq_per_fold,
    length_min=min_len,
    length_max=max_len,
    n_pics=n_switches,
    dtype=torch.float64
)

# Dataset loader
latch_task_loader = torch.utils.data.DataLoader(
    latch_task_dataset,
    batch_size=1,
    shuffle=False
)

# For each sample
for data_i, data in enumerate(latch_task_dataset):
    # Inputs and output
    data_inputs, data_outputs = data
# end for

