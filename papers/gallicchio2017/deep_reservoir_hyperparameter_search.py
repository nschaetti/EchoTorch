# -*- coding: utf-8 -*-
#
# File : papers/gallicchio2017/deep_reservoir_hyperparameters_search.py
# Description : Reproduction of the paper "Deep Reservoir Computing : A Critical Experiemental Analysis"
# (Gallicchio 2017)
# Date : 14th of September, 2020
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
import echotorch.utils
import echotorch.datasets as etds
import echotorch.transforms as ettr
import numpy as np
import echotorch.utils.optimization as optim
from papers.gallicchio2017.tools import evaluate_perturbations


# Function to test the ESN with specific hyper-parameters
def evaluation_function(parameters, datasets, n_samples=10):
    """
    Evaluate DeepESN on the task of perturbation of sequence of random symbols.
    :param parameters: Hyper-parameters used for evaluation.
    :param datasets: Datasets to use.
    :param n_samples: Number of times we repeat the evaluation.
    :return: The evaluation measure
    """
    # Get hyperparameters
    n_layers = parameters['n_layers']
    reservoir_size = parameters['reservoir_size']
    w_connectivity = parameters['w_connectivity']
    win_connectivity = parameters['win_connectivity']
    leak_rate = parameters['leak_rate']
    spectral_radius = parameters['spectral_radius']
    vocabulary_size = parameters['vocabulary_size']
    input_scaling = parameters['input_scaling']
    bias_scaling = parameters['bias_scaling']
    deep_esn_type = parameters['esn_type']
    n_samples_per_model = parameters['n_samples_per_model']
    sample_len = parameters['sample_len']
    perturbation_position = parameters['perturbation_position']
    use_cuda = parameters['use_cuda']
    dtype = parameters['dtype']

    # Perform experiment with the model
    for sample_i in range(n_samples):
        states_distances, KT, SF, TS = evaluate_perturbations(
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
            n_samples=n_samples_per_model,
            sample_len=sample_len,
            perturbation_position=perturbation_position,
            use_cuda=use_cuda,
            dtype=dtype
        )
    # end for

# end evaluation_function

# Exp. parameters
sample_len = 5000
n_samples = 10
vocabulary_size = 10
n_layers = 10
dtype = torch.float64

# Manual seed initialisation
echotorch.utils.manual_seed(1)

# Get a random optimizer
genetic_optimizer = optim.optimizer_factory.get_optimizer('genetic')

# Create the dataset
random_sequence_dataset = etds.TransformDataset(
    root_dataset=etds.RandomSymbolDataset(
        sample_len=sample_len,
        n_samples=n_samples,
        vocabulary_size=10
    ),
    transform=ettr.timeseries.ToOneHot(output_dim=vocabulary_size, dtype=dtype),
    transform_indices=[0]
)

# Parameters ranges
param_ranges = dict()
param_ranges['spectral_radius'] = np.linspace(0, 2.0, 1000)
param_ranges['leaky_rate'] = np.linspace(0.1, 1.0, 1000)
param_ranges['reservoir_size'] = np.arange(50, 510, 10)
param_ranges['connectivity'] = np.linspace(0.1, 1.0, 1000)
param_ranges['ridge_param'] = np.logspace(-10, 2, base=10, num=1000)
param_ranges['input_scaling'] = np.linspace(0.1, 1.0, 1000)
param_ranges['bias_scaling'] = np.linspace(0.0, 1.0, 1000)

# Launch the optimization of hyper-paramete
_, best_param, best_NRMSE = genetic_optimizer.optimize(
    evaluation_function,
    param_ranges,
    random_sequence_dataset,
    n_samples=5
)

# Show the result
print("Best hyper-parameters found : {}".format(best_param))
print("Best NRMSE : {}".format(best_NRMSE))