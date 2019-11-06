# -*- coding: utf-8 -*-
#
# File : papers/schaetti2016/ESNs-Based-RC-for-MNIST-Handwritten-Digits-Recognition.py
# Description : Echo State Networks-based Reservoir Computng for MNIST Handwritten Digits Recognition (schaetti 2016)
# Date : 6th of November, 2019
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
import torch.utils.data
import torchvision.datasets
import matplotlib.pyplot as plt


# Experiment parameters
reservoir_size = 500
spectral_radius = 0.99
leaky_rate = 0.5

# MNIST data set train
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        root=".",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    ),
    batch_size=1,
    shuffle=True
)

# MNIST data set test
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        root=".",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    ),
    batch_size=1,
    shuffle=True
)

# For each training sample
for batch_idx, (data, target) in enumerate(train_loader):
    print(data.shape)
    print(target)
    plt.imshow(data[0][0], cmap='gray')
    plt.show()
    if batch_idx == 10:
        break
    # end if
# end for
