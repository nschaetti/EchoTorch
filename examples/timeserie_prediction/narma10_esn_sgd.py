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
import torch.optim as optim
from echotorch.datasets.NARMADataset import NARMADataset
import echotorch.nn as etnn
import echotorch.utils
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import numpy as np
from echotorch.utils.matrix_generation import NormalMatrixGenerator
# import mdp
import matplotlib.pyplot as plt

# Parameters
spectral_radius = 0.9
leaky_rate = 1.0
learning_rate = 0.04
input_dim = 1
n_hidden = 100
n_iterations = 2000
train_sample_length = 5000
test_sample_length = 1000
n_train_samples = 1
n_test_samples = 1
batch_size = 1
momentum = 0.95
weight_decay = 0

# Use CUDA?
use_cuda = True
use_cuda = torch.cuda.is_available() if use_cuda else False

# Manual seed
# mdp.numx.random.seed(1)
np.random.seed(2)
torch.manual_seed(1)

# NARMA30 dataset
narma10_train_dataset = NARMADataset(train_sample_length, n_train_samples, system_order=10)
narma10_test_dataset = NARMADataset(test_sample_length, n_test_samples, system_order=10)

# Data loader
trainloader = DataLoader(narma10_train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = DataLoader(narma10_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ESN cell
# esn = etnn.LiESN(input_dim=input_dim, hidden_dim=n_hidden, output_dim=1, spectral_radius=spectral_radius, leaky_rate=0.9261, learning_algo='grad')
esn = etnn.ESN(input_dim=input_dim,
               hidden_dim=n_hidden,
               output_dim=1,
               w_generator=NormalMatrixGenerator(connectivity=0.1954, spetral_radius=0.9),
               win_generator=NormalMatrixGenerator(connectivity=0.1954, scale=0.9252, apply_spectral_radius=False),
               wbias_generator=NormalMatrixGenerator(connectivity=0.1954, scale=0.9252, apply_spectral_radius=False),
               learning_algo='grad')
if use_cuda:
    esn.cuda()
# end if

# Objective function
criterion = nn.MSELoss()

# Stochastic Gradient Descent
optimizer = optim.SGD(esn.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

# For each iteration
for epoch in range(n_iterations):
    # Iterate over batches
    for i, data in enumerate(trainloader):
        # Inputs and outputs
        inputs, targets = data
        inputs, targets = Variable(inputs), Variable(targets)
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()

        # Gradients to zero
        optimizer.zero_grad()

        # Forward
        out = esn(inputs)
        loss = criterion(out, targets)

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Print error measures
        print(f"Epoch: {epoch}/{n_iterations}...")
        print(f"Batch: {i}/{len(trainloader)}")
        print(f"Loss: {loss.item():.4f}")
        print(("Train MSE: {}".format(float(loss.data))))
        print(("Train NRMSE: {}".format(echotorch.utils.nrmse(out.data, targets.data))))
    # end for

    # Test reservoir
    dataiter = iter(testloader)
    test_u, test_y = next(dataiter)
    test_u, test_y = Variable(test_u), Variable(test_y)
    if use_cuda: test_u, test_y = test_u.cuda(), test_y.cuda()
    y_predicted = esn(test_u)

    # Print error measures
    print(("Test MSE: {}".format(echotorch.utils.mse(y_predicted.data, test_y.data))))
    print(("Test NRMSE: {}".format(echotorch.utils.nrmse(y_predicted.data, test_y.data))))
    print("")
# end for
