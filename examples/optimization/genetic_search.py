# -*- coding: utf-8 -*-
#
# File : echotorch/examples/optimization/grid_search.py
# Description : Optimize hyperparameters of an ESN with an exhaustive search.
# Date : 20 August, 2020
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
from echotorch.datasets.NARMADataset import NARMADataset
import echotorch.utils.optimization as optim
import numpy as np
from .narma_evaluation import evaluation_function

# Length of training samples
train_sample_length = 5000

# Length of test samples
test_sample_length = 1000

# How many training/test samples
n_train_samples = 1
n_test_samples = 1

# Manual seed initialisation
np.random.seed(1)
torch.manual_seed(1)

# Get a random optimizer
genetic_optimizer = optim.optimizer_factory.get_optimizer('genetic')

# NARMA10 dataset
narma10_train_dataset = NARMADataset(train_sample_length, n_train_samples, system_order=10)
narma10_test_dataset = NARMADataset(test_sample_length, n_test_samples, system_order=10)

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
    (narma10_train_dataset, narma10_test_dataset),
    n_samples=5
)

# Show the result
print("Best hyper-parameters found : {}".format(best_param))
print("Best NRMSE : {}".format(best_NRMSE))
