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
import torch.utils.data
from torch.autograd import Variable
import torchvision.datasets
import echotorch.utils.matrix_generation
import echotorch.nn.reservoir
import matplotlib.pyplot as plt


# Experiment parameters
reservoir_size = 500
spectral_radius = 0.99
leaky_rate = 0.5
batch_size = 8

# MNIST data set train
train_loader = torch.utils.data.DataLoader(
    echotorch.datasets.ImageToTimeseries(
        torchvision.datasets.MNIST(
            root=".",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        n_images=10
    ),
    batch_size=batch_size,
    shuffle=True
)

# MNIST data set test
test_loader = torch.utils.data.DataLoader(
    echotorch.datasets.ImageToTimeseries(
        torchvision.datasets.MNIST(
            root=".",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        n_images=10
    ),
    batch_size=batch_size,
    shuffle=True
)


# Tranpose class index to one-hot vector
def to_one_hot(x):
    """
    Transpose class index to one-hot vector
    :param x:
    :return:
    """
    x = x.long()
    return torch.eye(10).index_select(0, x.data)
# end to_one_hot


# Matrices generator
"""w_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(connectivity=0.1)
win_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(connectivity=0.1)
wbias_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(connectivity=0.1)

# Instanciate ESN module
esn = echotorch.nn.reservoir.LiESN(
    input_dim=28,
    hidden_dim=reservoir_size,
    output_dim=10,
    leaky_rate=0.5,
    spectral_radius=0.99,
    input_scaling=0.5,
    w_generator=w_generator,
    win_generator=win_generator,
    wbias_generator=wbias_generator
)"""

# For each training sample
for batch_idx, (data, targets) in enumerate(train_loader):
    # Remove channel
    data = data.reshape(batch_size, 280, 28)

    # To Variable
    inputs, targets = Variable(data, requires_grad=False), Variable(targets, requires_grad=False)

    # Feed ESN
    # esn(inputs, targets)
# end for

# Finish training
# esn.finalize()
