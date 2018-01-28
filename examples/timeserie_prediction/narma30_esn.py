# -*- coding: utf-8 -*-
#
# File : examples/SwitchAttractor/switch_attractor_esn
# Description : Attractor switching task with ESN.
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
import EchoTorch.tools
from EchoTorch.datasets.NARMADataset import NARMADataset
import EchoTorch.nn as etnn
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Manual seed
np.random.seed(2)
torch.manual_seed(1)

# Dataset params
sample_length = 500
n_train_samples = 640
n_test_samples = 5
batch_size = 64

# Spectral radius
spectral_radius = 0.9

# NARMA30 dataset
narma30_train_dataset = NARMADataset(sample_length, n_train_samples, system_order=30, seed=1)
narma30_test_dataset = NARMADataset(sample_length, n_test_samples, system_order=30, seed=10)

# Data loader
trainloader = DataLoader(narma30_train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = DataLoader(narma30_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ESN properties
input_dim = 1
n_hidden = 20

# ESN cell
esn = etnn.ESNCell(input_dim, n_hidden, spectral_radius=spectral_radius)
print(u"Spectral radius : {}".format(esn.get_spectral_radius()))

# Linear layer
linear = nn.Linear(n_hidden, 1, bias=True)

# Objective function
criterion = nn.MSELoss()

# Learning rate
learning_rate = 0.001

# Stochastic Gradient Descent
optimizer = optim.SGD(linear.parameters(), lr=learning_rate, momentum=0, weight_decay=0.1)

# Number of iterations
n_iterations = 20

# Losses
losses = torch.zeros(n_iterations * (n_train_samples / batch_size))
sample_pos = 0

# For each iteration
for epoch in range(n_iterations):
    for data in trainloader:
        # Zero loss
        loss = 0

        # Reset gradient
        optimizer.zero_grad()

        # For each sample
        for i_sample in range(data[0].size()[0]):
            # Inputs and outputs
            inputs, targets = data[0][i_sample], data[1][i_sample]
            inputs, targets = Variable(inputs), Variable(targets)

            # Init hidden
            hidden = esn.init_hidden()

            # Get hidden states
            hidden_states = esn(inputs, hidden)

            # Linear output
            out = linear(hidden_states)

            # Loss
            loss += criterion(out, targets)
        # end for

        # Backward pass
        loss.backward()

        # Save loss and gradients
        losses[sample_pos] = float(loss.data)

        # Update weights
        optimizer.step()

        # Print
        print(u"Iteration {}, Batch {}, loss {}".format(epoch, sample_pos, loss.data[0]))

        # Next sample
        sample_pos += 1
    # end for

    # Print
    print(u"Iteration {}, loss {}".format(epoch, loss.data[0]))
# end for

# Show evolution of losses
plt.plot(losses.numpy(), c='r')
plt.show()
