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
n_layers = [10, 1, 10, 10]
reservoir_size = [100, 1000, 100, 100]
w_connectivity = [0.25, 0.25, 0.25, 0.25]
win_connectivity = [0.1, 0.1, 0.1, 0.1]
leak_rate = [0.55, 0.55, 0.55, 0.55]
spectral_radius = [0.9, 0.9, 0.9, 0.9]
vocabulary_size = 10
input_scaling = [1.0, 1.0, 1.0, 1.0]
bias_scaling = [0.0, 0.0, 0.0, 0.0]
input_dim = vocabulary_size
deep_esn_type = ['IF', 'esn', 'IA', 'GE']
n_samples = 10
sample_len = 5000
perturbation_position = 100
plot_length = 500
dtype=torch.float64
use_cuda = False and torch.cuda.is_available()

# Perform experiment with Deep-ESN (Input-to-First)
desn_if_states_distances, desn_if_tau, desn_if_sp_rule, desn_if_ts_separation = evaluate_perturbations(
    n_layers=n_layers[0],
    reservoir_size=reservoir_size[0],
    w_connectivity=w_connectivity[0],
    win_connectivity=win_connectivity[0],
    leak_rate=leak_rate[0],
    spectral_radius=spectral_radius[0],
    vocabulary_size=vocabulary_size,
    input_scaling=input_scaling[0],
    bias_scaling=bias_scaling[0],
    esn_type=deep_esn_type[0],
    n_samples=n_samples,
    sample_len=sample_len,
    perturbation_position=perturbation_position,
    use_cuda=use_cuda,
    dtype=dtype
)

# Perform experiment with Deep-ESN (Input-to-All)
desn_ia_states_distances, desn_ia_tau, desn_ia_sp_rule, desn_ia_ts_separation = evaluate_perturbations(
    n_layers=n_layers[2],
    reservoir_size=reservoir_size[2],
    w_connectivity=w_connectivity[2],
    win_connectivity=win_connectivity[2],
    leak_rate=leak_rate[2],
    spectral_radius=spectral_radius[2],
    vocabulary_size=vocabulary_size,
    input_scaling=input_scaling[2],
    bias_scaling=bias_scaling[2],
    esn_type=deep_esn_type[2],
    n_samples=n_samples,
    sample_len=sample_len,
    perturbation_position=perturbation_position,
    use_cuda=use_cuda,
    dtype=dtype
)

# Perform experiment with Grouped-ESNs (No connections)
desn_ge_states_distances, desn_ge_tau, desn_ge_sp_rule, desn_ge_ts_separation = evaluate_perturbations(
    n_layers=n_layers[3],
    reservoir_size=reservoir_size[3],
    w_connectivity=w_connectivity[3],
    win_connectivity=win_connectivity[3],
    leak_rate=leak_rate[3],
    spectral_radius=spectral_radius[3],
    vocabulary_size=vocabulary_size,
    input_scaling=input_scaling[3],
    bias_scaling=bias_scaling[3],
    esn_type=deep_esn_type[3],
    n_samples=n_samples,
    sample_len=sample_len,
    perturbation_position=perturbation_position,
    use_cuda=use_cuda,
    dtype=dtype
)

# Perform experiment with ESN
esn_states_distances, esn_tau, esn_sp_rule, esn_ts_separation = evaluate_perturbations(
    n_layers=n_layers[1],
    reservoir_size=reservoir_size[1],
    w_connectivity=w_connectivity[1],
    win_connectivity=win_connectivity[1],
    leak_rate=leak_rate[1],
    spectral_radius=spectral_radius[1],
    vocabulary_size=vocabulary_size,
    input_scaling=input_scaling[1],
    bias_scaling=bias_scaling[1],
    esn_type=deep_esn_type[1],
    n_samples=n_samples,
    sample_len=sample_len,
    perturbation_position=perturbation_position,
    use_cuda=use_cuda,
    dtype=dtype
)

# Plot result from IF and ESN
plt.figure(figsize=(8, 6))
plt.title("DeepESN-IF and Shallow ESN")
plt.xlabel("Time Step")
plt.ylabel("Distance between States")
for layer_i in range(n_layers[0]):
    plt.plot(desn_if_states_distances[:plot_length, layer_i], linestyle='-', color=(0.0, 0.0, 1.0, (1.0/n_layers[0]) * layer_i))
# end for
plt.plot(esn_states_distances[:plot_length, 0], linestyle='--', color='r')
plt.show()

# Plot result from IA and ESN
plt.figure(figsize=(8, 6))
plt.title("DeepESN-IA and Shallow ESN")
plt.xlabel("Time Step")
plt.ylabel("Distance between States")
for layer_i in range(n_layers[0]):
    plt.plot(desn_ia_states_distances[:plot_length, layer_i], linestyle='-', color=(0.0, 0.0, 1.0, (1.0/n_layers[0]) * layer_i))
# end for
plt.plot(esn_states_distances[:plot_length, 0], linestyle='--', color='r')
plt.show()

# Plot result from GE and ESN
plt.figure(figsize=(8, 6))
plt.title("Grouped-ESN and Shallow ESN")
plt.xlabel("Time Step")
plt.ylabel("Distance between States")
for layer_i in range(n_layers[0]):
    plt.plot(desn_ge_states_distances[:plot_length, layer_i], linestyle='-', color=(0.0, 0.0, 1.0, (1.0/n_layers[0]) * layer_i))
# end for
plt.plot(esn_states_distances[:plot_length, 0], linestyle='--', color='r')
plt.show()

# Plot tau for each variant
plt.figure(figsize=(8, 6))
plt.title("Tau")
plt.bar(np.arange(4), [desn_if_tau, desn_ia_tau, desn_ge_tau, esn_tau])
plt.xticks(np.arange(4), ('IF', 'IA', 'GE', 'ESN'))
plt.show()

# Plot tau for each variant
plt.figure(figsize=(8, 6))
plt.title("Spearman's Rule")
plt.bar(np.arange(4), [desn_if_sp_rule, desn_ia_sp_rule, desn_ge_sp_rule, esn_sp_rule])
plt.xticks(np.arange(4), ('IF', 'IA', 'GE', 'ESN'))
plt.show()

# Plot tau for each variant
plt.figure(figsize=(8, 6))
plt.title("Timescale separation")
plt.bar(np.arange(4), [desn_if_ts_separation, desn_ia_ts_separation, desn_ge_ts_separation, esn_ts_separation])
plt.xticks(np.arange(4), ('IF', 'IA', 'GE', 'ESN'))
plt.show()
