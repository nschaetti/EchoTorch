# -*- coding: utf-8 -*-
#
# File : examples/timeserie_prediction/narma10_esn
# Description : NARMA-10 prediction with ESN.
# Date : 26th of January, 2018
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
import torch
from echotorch.datasets.NARMADataset import NARMADataset
import echotorch.nn.conceptors as ecc
import echotorch.utils.matrix_generation as mg
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import numpy as np

# Length of training samples
train_sample_length = 5

# Length of test samples
test_sample_length = 2

# How many training/test samples
n_train_samples = 1
n_test_samples = 1

# Batch size (how many sample processed at the same time?)
batch_size = 1

# Reservoir hyper-parameters
spectral_radius = 0.99
leaky_rate = 1.0
input_dim = 1
reservoir_size = 100
connectivity = 0.1
ridge_param = 0.0000001
w_ridge_param = 0.0001

# Predicted/target plot length
plot_length = 200

# Use CUDA?
use_cuda = False
use_cuda = torch.cuda.is_available() if use_cuda else False

# Manual seed initialisation
np.random.seed(1)
torch.manual_seed(1)

# NARMA30 dataset
narma10_train_dataset = NARMADataset(train_sample_length, n_train_samples, system_order=10, seed=1)
narma10_test_dataset = NARMADataset(test_sample_length, n_test_samples, system_order=10, seed=10)

# Data loader
trainloader = DataLoader(narma10_train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = DataLoader(narma10_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Get matrix generators
matrix_generator = mg.matrix_factory.get_generator(
    name='normal',
    connectivity=0.1,
    mean=0.0,
    std=1.0
)

# Create a Leaky-integrated ESN,
# with least-square training algo.
# esn = etrs.ESN(
spesn = ecc.SPESN(
    input_dim=input_dim,
    hidden_dim=reservoir_size,
    output_dim=1,
    spectral_radius=spectral_radius,
    learning_algo='inv',
    w_generator=matrix_generator,
    win_generator=matrix_generator,
    wbias_generator=matrix_generator,
    input_scaling=1.0,
    bias_scaling=0,
    ridge_param=ridge_param,
    w_ridge_param=w_ridge_param
)

# Transfer in the GPU if possible
if use_cuda:
    spesn.cuda()
# end if

# For each batch
for data in trainloader:
    # Inputs and outputs
    inputs, targets = data

    # Transform data to Variables
    inputs, targets = Variable(inputs), Variable(targets)
    if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()

    # ESN need inputs and targets
    spesn(inputs, targets)
# end for

# Now we finalize the training by
# computing the output matrix Wout.
spesn.finalize()
