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
                echotorch.transforms.images.Concat([
                    echotorch.transforms.images.CropResize(size=15),
                    torchvision.transforms.Compose([
                        echotorch.transforms.images.Rotate(degree=30),
                        echotorch.transforms.images.CropResize(size=15)
                    ]),
                    torchvision.transforms.Compose([
                        echotorch.transforms.images.Rotate(degree=60),
                        echotorch.transforms.images.CropResize(size=15)
                    ]),
                    torchvision.transforms.Compose([
                        echotorch.transforms.images.Rotate(degree=60),
                        echotorch.transforms.images.CropResize(size=15)
                    ])
                ],
                    sequential=True
                ),
                torchvision.transforms.ToTensor()
            ]),
            target_transform=echotorch.transforms.targets.ToOneHot(class_size=10)
        ),
        n_images=10
    ),
    batch_size=batch_size,
    shuffle=False
)

# MNIST data set test
test_loader = torch.utils.data.DataLoader(
    echotorch.datasets.ImageToTimeseries(
        torchvision.datasets.MNIST(
            root=".",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                echotorch.transforms.images.Concat([
                    echotorch.transforms.images.CropResize(size=15),
                    torchvision.transforms.Compose([
                        echotorch.transforms.images.Rotate(degree=30),
                        echotorch.transforms.images.CropResize(size=15)
                    ]),
                    torchvision.transforms.Compose([
                        echotorch.transforms.images.Rotate(degree=60),
                        echotorch.transforms.images.CropResize(size=15)
                    ]),
                    torchvision.transforms.Compose([
                        echotorch.transforms.images.Rotate(degree=60),
                        echotorch.transforms.images.CropResize(size=15)
                    ])
                ],
                    sequential=True
                ),
                torchvision.transforms.ToTensor()
            ]),
            target_transform=echotorch.transforms.targets.ToOneHot(class_size=10)
        ),
        n_images=10
    ),
    batch_size=batch_size,
    shuffle=False
)

# Matrices generator
"""
# New ESN-JS module
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
    print(data.size())
    # Remove channel
    data = data.reshape(batch_size, 150, 60)

    # To Variable
    inputs, targets = Variable(data, requires_grad=False), Variable(targets, requires_grad=False)

    # Plot
    plt.imshow(data[0].t(), cmap='gray')
    plt.show()
    if batch_idx == 1:
        break
    # end if
    # Feed ESN
    # esn(inputs, targets)
# end for

# Finish training
# esn.finalize()
