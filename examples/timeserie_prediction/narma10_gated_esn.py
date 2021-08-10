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

# Parameters
spectral_radius = 0.9
leaky_rate = 1.0
learning_rate = 0.04
reservoir_dim = 100
hidden_dim = 20
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
echotorch.utils.manual_seed(1)

# NARMA30 dataset
narma10_train_dataset = NARMADataset(train_sample_length, n_train_samples, system_order=10, seed=1)
narma10_test_dataset = NARMADataset(test_sample_length, n_test_samples, system_order=10, seed=10)

# Data loader
trainloader = DataLoader(narma10_train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = DataLoader(narma10_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Linear output
linear = nn.Linear(in_features=hidden_dim, out_features=1)

# ESN cell
gated_esn = etnn.GatedESN(
    input_dim=input_dim,
    reservoir_dim=reservoir_dim,
    pca_dim=hidden_dim,
    hidden_dim=hidden_dim,
    leaky_rate=leaky_rate,
    spectral_radius=spectral_radius
)
if use_cuda:
    gated_esn.cuda()
    linear.cuda()
# end if

# Objective function
criterion = nn.MSELoss()

# Stochastic Gradient Descent
optimizer = optim.SGD(gated_esn.parameters(), lr=learning_rate, momentum=momentum)

# For each iteration
for epoch in range(n_iterations):
    # Iterate over batches
    for data in trainloader:
        # Inputs and outputs
        inputs, targets = data
        inputs, targets = Variable(inputs), Variable(targets)
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
        print(inputs)
        print(targets)
        # Gradients to zero
        optimizer.zero_grad()

        # Forward
        out = linear(gated_esn(inputs))
        print(out)
        exit()
        loss = criterion(out, targets)

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Print error measures
        print(u"Train MSE: {}".format(float(loss.data)))
        print(u"Train NRMSE: {}".format(echotorch.utils.nrmse(out.data, targets.data)))
    # end for

    # Test reservoir
    dataiter = iter(testloader)
    test_u, test_y = dataiter.next()
    test_u, test_y = Variable(test_u), Variable(test_y)
    if use_cuda: test_u, test_y = test_u.cuda(), test_y.cuda()
    y_predicted = gated_esn(test_u)

    # Print error measures
    print(u"Test MSE: {}".format(echotorch.utils.mse(y_predicted.data, test_y.data)))
    print(u"Test NRMSE: {}".format(echotorch.utils.nrmse(y_predicted.data, test_y.data)))
    print(u"")
# end for
