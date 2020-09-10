# -*- coding: utf-8 -*-
#
# File : papers/gallicchio2017/deep_reservoir_computing_critical_experiemental_analysis.py
# Description : Reproduction of the paper "Deep Reservoir Computing : A Critical Experiemental Analysis"
# (Gallicchio 2017)
# Date : 10th of September, 2020
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
# Copyright Nils Schaetti <nils.schaetti@unine.ch>, <nils.schaetti@unige.ch>

# Imports
import torch
import torch.utils.data
import echotorch.nn as etnn
import echotorch.datasets as etds
from echotorch.utils.matrix_generation import matrix_factory


# Experiment parameters
n_layers = 10
reservoir_size = 100
leak_rate = 0.55
spectral_radius = 0.9
vocabulary_size = 10
input_scaling = 1.0
input_dim = vocabulary_size
deep_esn_type = 'IF'
sample_len = 5000
perturbation_position = 100
use_cuda = False and torch.cuda.is_available()

# Dataset generating sequences of random symbols
random_symbols_loader = torch.utils.data.DataLoader(
    etds.RandomSymbolDataset(
        sample_len=sample_len,
        n_samples=100,
        vocabulary_size=10
    )
)

# Generator for internal matrices W
w_generator = matrix_factory.get_generator(
    name='normal',
    spectral_radius=spectral_radius
)

# Generator for inputs-to-reservoir matrices Win
win_generator = matrix_factory.get_generator(
    name='normal'
)

# Generator for internal units biases
wbias_generator = matrix_factory.get_generator(
    name='normal'
)

# Deep ESN
deep_esn = etnn.reservoir.DeepESN(
    n_layers=n_layers,
    input_dim=input_dim,
    hidden_dim=reservoir_size,
    output_dim=1,
    w_generator=w_generator,
    win_generator=win_generator,
    wbias_generator=wbias_generator,
    input_scaling=input_scaling,
    input_type=deep_esn_type,
    dtype=torch.float64
)

# Show the model
print(deep_esn)

# Use cuda ?
if use_cuda:
    deep_esn.cuda()
# end if

for batch_idx, (data, targets) in enumerate(train_loader):
    pass
# end for