# -*- coding: utf-8 -*-
#
# File : examples/datasets/triplet_batching.py
# Description : Take a dataset with different classes and create a dataset of triplets with an anchor (A) and positive
# example (same class) and a negative one (different class).
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
import echotorch.datasets as etds
import torch
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import random

# Random seed
random.seed(1)

# Dataset params
train_sample_length = 1000
n_train_samples = 1
batch_size = 1

# First pattern
pattern1_training = etds.SinusoidalTimeseries(
    sample_len=train_sample_length,
    n_samples=n_train_samples,
    a=1,
    period=0.711572515,
    start=11
)

# Second pattern
pattern2_training = etds.SinusoidalTimeseries(
    sample_len=train_sample_length,
    n_samples=n_train_samples,
    a=1,
    period=0.63918467,
    start=0
)

# Third pattern
pattern3_training = etds.PeriodicSignalDataset(
    sample_len=train_sample_length,
    n_samples=n_train_samples,
    period=[-1.0, -1.0, -0.1, 0.2, 1.0]
)

# Fourth pattern
pattern4_training = etds.PeriodicSignalDataset(
    sample_len=train_sample_length,
    n_samples=n_train_samples,
    period=[-1.0, -0.8, -0.1, 0.2, 1.0]
)

# Composer
dataset_training = etds.DatasetComposer([pattern1_training, pattern2_training, pattern3_training, pattern4_training])

# Sequence of timeseries
dataset_sequence_training = etds.TimeseriesBatchSequencesDataset(
    root_dataset=dataset_training,
    window_size=100,
    data_indices=[0, 1],
    stride=100,
    time_axis=0,
    dataset_in_memory=True
)

# Triplet batching dataset
dataset_triplet_batching_sequence = etds.TripletBatching(
    root_dataset=dataset_sequence_training,
    data_index=0,
    target_index=2,
    target_count=4,
    n_samples=10,
    target_type='tensor'
)

# Data loader
dataset_triplet_batching_sequence_loader = DataLoader(
    dataset_triplet_batching_sequence,
    batch_size=batch_size,
    shuffle=False,
    num_workers=1
)

# Go through all samples
for data in dataset_triplet_batching_sequence_loader:
    # Data
    anchor_sample, positive_sample, negative_sample = data

    # Classes
    anchor_class = anchor_sample[2].item()
    positive_class = positive_sample[2].item()
    negative_class = negative_sample[2].item()

    # Print classes
    print("Anchor: {}, Positive: {}, Negative: {}".format(
        anchor_class,
        positive_class,
        negative_class
    ))

    # Plot
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title("Anchor ({})".format(anchor_class))
    plt.plot(anchor_sample[0][0].numpy(), 'b')
    plt.subplot(3, 1, 2)
    plt.title("Positive ({})".format(positive_class))
    plt.plot(positive_sample[0][0].numpy(), 'g')
    plt.subplot(3, 1, 3)
    plt.title("Negative ({})".format(negative_class))
    plt.plot(negative_sample[0][0].numpy(), 'r')
    plt.show()
# end for
