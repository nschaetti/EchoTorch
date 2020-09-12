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

# Experiment parameters
n_layers = 1
reservoir_size = 1000
w_connectivity = 0.25
win_connectivity = 0.25
leak_rate = 0.55
spectral_radius = 0.9
vocabulary_size = 10
input_scaling = 1.0
bias_scaling = 0.0
input_dim = vocabulary_size
deep_esn_type = 'IF'
n_samples = 2
sample_len = 5000
perturbation_position = 100
plot_length = 500
plot_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b']
plot_line_types = ['-', '-', '-', '-', '-', '-', '-', '--', '--', '--']
dtype=torch.float64
use_cuda = False and torch.cuda.is_available()

# Dataset generating sequences of random symbols
random_symbols_loader = torch.utils.data.DataLoader(
    etds.TransformDataset(
        root_dataset=etds.RandomSymbolDataset(
            sample_len=sample_len,
            n_samples=n_samples,
            vocabulary_size=10
        ),
        transform=ettr.timeseries.ToOneHot(output_dim=vocabulary_size, dtype=dtype),
        transform_indices=[0]
    )
)

# Generator for internal matrices W
w_generator = matrix_factory.get_generator(
    name='normal',
    spectral_radius=spectral_radius,
    connectivity=w_connectivity
)

# Generator for inputs-to-reservoir matrices Win
win_generator = matrix_factory.get_generator(
    name='normal',
    connectivity=win_connectivity
)

# Generator for internal units biases
wbias_generator = matrix_factory.get_generator(
    name='normal',
    scale=bias_scaling
)

# Deep ESN
esn = etnn.reservoir.LiESN(
    input_dim=input_dim,
    hidden_dim=reservoir_size,
    output_dim=vocabulary_size,
    leaky_rate=leak_rate,
    w_generator=w_generator,
    win_generator=win_generator,
    wbias_generator=wbias_generator,
    input_scaling=input_scaling,
    dtype=dtype
)

# Show the model
print(esn)

# Use cuda ?
if use_cuda:
    esn.cuda()
# end if

# Go through all the dataset
for batch_idx, data in enumerate(random_symbols_loader):
    # Data
    unperturbed_input = data[0]

    # Perturb sequence at the right position
    perturbed_input = unperturbed_input.clone()
    perturbed_input[0, perturbation_position] = torch.zeros(vocabulary_size)
    perturbed_input[0, perturbation_position, torch.randint(vocabulary_size, size=(1, 1)).item()] = 1.0

    # To variables
    unperturbed_input, perturbed_input = Variable(unperturbed_input.double()), Variable(perturbed_input.double())

    # CUDA
    if use_cuda:
        unperturbed_input = unperturbed_input.cuda()
        perturbed_input = perturbed_input.cuda()
    # end if

    # Feed both version to the DeepESN
    unperturbed_states = esn(unperturbed_input, unperturbed_input)
    perturbed_states = esn(perturbed_input, perturbed_input)

    # New figure
    plt.figure(figsize=(10, 8))

    # Compute euclidian distance between each state for each layer
    states_distances = euclidian_distances(
        unperturbed_states,
        perturbed_states,
        n_layers
    )

    # Plot distances
    plt.plot(
        states_distances[0, perturbation_position:perturbation_position+plot_length, 0].numpy(),
        color='r',
        linestyle='-'
    )

    # Show the plot
    plt.show()

    # Perturbation effect
    P = perturbation_effect(states_distances[:, perturbation_position:])
    print("Layer perturbation durations : {}".format(P.item()))
# end for


