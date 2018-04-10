# -*- coding: utf-8 -*-
#
# File : examples/timeserie_prediction/switch_attractor_esn
# Description : NARMA 30 prediction with ESN.
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
import echotorch.nn as etnn
import echotorch.utils
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import numpy as np
import mdp

# Dataset params
train_sample_length = 5000
test_sample_length = 1000
n_train_samples = 1
n_test_samples = 1
batch_size = 1
spectral_radius = 0.9
leaky_rates = [1.0, 0.5, 0.1]
input_dim = 1
n_hidden = [100, 100, 100]

# Use CUDA?
use_cuda = False
use_cuda = torch.cuda.is_available() if use_cuda else False

# Manual seed
mdp.numx.random.seed(1)
np.random.seed(2)
torch.manual_seed(1)

# NARMA30 dataset
narma10_train_dataset = NARMADataset(train_sample_length, n_train_samples, system_order=10, seed=1)
narma10_test_dataset = NARMADataset(test_sample_length, n_test_samples, system_order=10, seed=10)

# Data loader
trainloader = DataLoader(narma10_train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = DataLoader(narma10_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ESN cell
esn = etnn.StackedESN(input_dim=input_dim, hidden_dim=n_hidden, output_dim=1, spectral_radius=spectral_radius, learning_algo='inv', leaky_rate=leaky_rates)

# For each batch
for data in trainloader:
    # Inputs and outputs
    inputs, targets = data

    # To variable
    inputs, targets = Variable(inputs), Variable(targets)
    if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()

    # Accumulate xTx and xTy
    hidden_states = esn(inputs, targets)
    for i in range(10):
        print(hidden_states[0, i])
    # end if
# end for