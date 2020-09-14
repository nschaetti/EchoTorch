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
import echotorch.transforms as ettr
from echotorch.utils.matrix_generation import matrix_factory
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from papers.gallicchio2017.tools import euclidian_distances, perturbation_effect, ranking_of_layers, kendalls_tau, \
    spearmans_rule, timescales_separation
from papers.gallicchio2017.tools import evaluate_perturbations

# Experiment parameters
n_layers = 10
reservoir_size = 100
w_connectivity = 0.25
win_connectivity = 0.1
leak_rate = 0.55
spectral_radius = 0.9
vocabulary_size = 10
input_scaling = 1.0
bias_scaling = 0.0
input_dim = vocabulary_size
deep_esn_type = 'IF'
n_samples = 1
sample_len = 5000
perturbation_position = 100
plot_length = 500
plot_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b']
plot_line_types = ['-', '-', '-', '-', '-', '-', '-', '--', '--', '--']
dtype=torch.float64
use_cuda = False and torch.cuda.is_available()

# Perform experiment with Deep-ESN (Input-to-First)
measure_list_deepesn_if = evaluate_perturbations(
    n_layers=n_layers,
    reservoir_size=reservoir_size,
    w_connectivity=w_connectivity,
    win_connectivity=win_connectivity,
    leak_rate=leak_rate,
    spectral_radius=spectral_radius,
    vocabulary_size=vocabulary_size,
    input_scaling=input_scaling,
    bias_scaling=bias_scaling,
    esn_type=deep_esn_type,
    n_samples=n_samples,
    sample_len=sample_len,
    perturbation_position=perturbation_position,
    use_cuda=use_cuda,
    dtype=dtype
)

print(measure_list_deepesn_if)
