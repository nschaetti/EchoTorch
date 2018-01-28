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
from EchoTorch.datasets.MemTestDataset import MemTestDataset
import EchoTorch.nn as etnn
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

# Dataset params
sample_length = 20
n_samples = 2
batch_size = 5

# MemTest dataset
memtest_dataset = MemTestDataset(sample_length, n_samples, seed=1)

# Data loader
dataloader = DataLoader(memtest_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ESN properties
input_dim = 1
n_hidden = 20

# ESN cell
esn = etnn.ESNCell(input_dim, n_hidden)

# Linear layer
linear = nn.Linear(n_hidden, 1)

# Objective function
criterion = nn.MSELoss()

# Learning rate
learning_rate = 0.0001

# Number of iterations
n_iterations = 10

for data in dataloader:
    # For each sample
    for i_sample in range(data[0].size()[0]):
        # Inputs and outputs
        inputs, outputs = data[0][i_sample], data[1][i_sample]
        inputs, outputs = Variable(inputs), Variable(outputs)

        # Show the graph
        plt.plot(inputs.data.numpy(), c='b')
        plt.plot(outputs.data[:, 9].numpy(), c='r')
        plt.show()
    # end for
# end for