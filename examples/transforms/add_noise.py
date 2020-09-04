# -*- coding: utf-8 -*-
#
# File : examples/transforms/add_noise.py
# Description : Add noise to a timeseries
# Date : 21th of August, 2020
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
import echotorch.datasets as etds
import echotorch.transforms as ettr
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt


# A dataset of sine patterns
sine_dataset = etds.SinusoidalTimeseries(sample_len=1000, n_samples=1, period=9.3547)

# Noisy sine dataset
sine_noisy_dataset = etds.TransformDataset(
    root_dataset=sine_dataset,
    transform=ettr.timeseries.AddNoise(input_dim=1, mu=torch.tensor([0.0]), std=torch.tensor([0.25]))
)

# Timeseries batch sequence dataset
sine_sequence_dataset = etds.TimeseriesBatchSequencesDataset(
    root_dataset=sine_noisy_dataset,
    window_size=100,
    stride=100,
    data_indices=None,
    dataset_in_memory=True
)

# Data loader
sine_sequence_dataset_loader = DataLoader(sine_sequence_dataset, batch_size=1, shuffle=False)

# Go throuth all samples
for data in sine_sequence_dataset_loader:
    # Plot
    plt.plot(data[0].numpy(), 'b')
    plt.show()
# end for
