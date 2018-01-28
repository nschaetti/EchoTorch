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
from EchoTorch.datasets.MackeyGlassDataset import MackeyGlassDataset
import EchoTorch.nn as etnn
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

# Dataset params
sample_length = 1000
n_samples = 40
batch_size = 5

# Mackey glass dataset
mackey_glass_dataset = MackeyGlassDataset(sample_length, n_samples, tau=30)

# Data loader
dataloader = DataLoader(mackey_glass_dataset, batch_size=5, shuffle=False, num_workers=2)

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

# For each iterations
for i_iter in range(n_iterations):
    # Iterate through batches
    for i_batch, sample_batched in enumerate(dataloader):
        # For each sample
        for i_sample in range(sample_batched.size()[0]):
            # Inputs and outputs
            inputs = Variable(sample_batched[i_sample][:-1], requires_grad=False)
            outputs = Variable(sample_batched[i_sample][1:], requires_grad=False)
            esn_outputs = torch.zeros(sample_length-1, 1)
            gradients = torch.zeros(sample_length-1, 1)

            # Init hidden
            hidden = esn.init_hidden()

            # Zero grad
            esn.zero_grad()

            # Null loss
            loss = 0

            # For each input
            for pos in range(sample_length-1):
                # Compute next state
                next_hidden = esn(inputs[pos], hidden)

                # Linear output
                out = linear(next_hidden)
                esn_outputs[pos, :] = out.data

                # Add loss
                loss += criterion(out, outputs[pos])
            # end for

            # Loss
            loss.div_(sample_length-1)

            loss.backward()

            # Update parameters
            for p in linear.parameters():
                p.data.add_(-learning_rate, p.grad.data)
            # end for

            # Show the graph only for last sample of iteration
            #if i_batch == len(dataloader) - 1 and i_sample == len(sample_batched) -1 :
            """plt.plot(inputs.data.numpy(), c='b')
            plt.plot(outputs.data.numpy(), c='lightblue')
            plt.plot(esn_outputs.numpy(), c='r')
            plt.show()"""
            # end if
        # end for
    # end for

    # Print
    print(u"Iteration {}, loss {}".format(i_iter, loss.data[0]))
# end for
