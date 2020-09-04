# -*- coding: utf-8 -*-
#
# File : examples/datasets/latch_copy_repeat.py
# Description : Generate data for three common tasks : latch, copy and repeat.
# Date : 21th of July, 2020
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
import torch.utils.data
import echotorch.nn.reservoir
import echotorch.datasets as etda
import matplotlib.pyplot as plt

# Latch parameters
min_len = 50
max_len = 200
n_switches = 3

# Init. random number generators
echotorch.utils.manual_seed(1)

# Copy task dataset
copy_task_dataset = etda.CopyTaskDataset(
    n_samples=1,
    length_min=1,
    length_max=20,
    n_inputs=8,
    dtype=torch.float64
)

# Latch task dataset
latch_task_dataset = etda.LatchTaskDataset(
    n_samples=1,
    length_min=min_len,
    length_max=max_len,
    n_pics=n_switches,
    dtype=torch.float64
)

# Repeat task dataset
repeat_task_dataset = etda.RepeatTaskDataset(
    n_samples=1,
    length_min=1,
    length_max=20,
    n_inputs=8,
    min_repeat=3,
    max_repeat=5,
    dtype=torch.float64
)

# Dataset loader
copy_task_loader = torch.utils.data.DataLoader(copy_task_dataset, batch_size=1, shuffle=False)
latch_task_loader = torch.utils.data.DataLoader(latch_task_dataset, batch_size=1, shuffle=False)
repeat_task_loader = torch.utils.data.DataLoader(repeat_task_dataset, batch_size=1, shuffle=False)

# Show the copy task dataset
for data_i, data in enumerate(copy_task_loader):
    # Inputs and output
    data_inputs, data_outputs = data

    # Plot inputs and output
    plt.figure(figsize=(6, 6))
    plt.subplot(2, 1, 1)
    plt.title("Copy inputs")
    plt.imshow(data_inputs[0].t().numpy(), cmap='Greys')
    plt.subplot(2, 1, 2)
    plt.title("Copy outputs")
    plt.imshow(data_outputs[0].t().numpy(), cmap='Greys')
    plt.show()
# end for

# Close
plt.close()

# Show the repeat task dataset
for data_i, data in enumerate(repeat_task_loader):
    # Inputs and output
    data_inputs, data_outputs = data

    # Plot inputs and output
    plt.figure(figsize=(6, 6))
    plt.subplot(2, 1, 1)
    plt.title("Repeat inputs")
    plt.imshow(data_inputs[0].t().numpy(), cmap='Greys')
    plt.subplot(2, 1, 2)
    plt.title("Repeat outputs")
    plt.imshow(data_outputs[0].t().numpy(), cmap='Greys')
    plt.show()
# end for

# Close
plt.close()

# For each sample
for data_i, data in enumerate(latch_task_dataset):
    # Inputs and output
    data_inputs, data_outputs = data

    # Plot
    plt.title("Latch task")
    plt.plot(data_inputs[0].numpy(), 'b')
    plt.plot(data_outputs[0].numpy(), 'r')
    plt.show()
# end for

# Close
plt.close()
