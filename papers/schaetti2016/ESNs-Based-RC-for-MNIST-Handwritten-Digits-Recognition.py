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
import sys
import torch.utils.data
from torch.autograd import Variable
import torchvision.datasets
import echotorch.nn.reservoir
import matplotlib.pyplot as plt
from tqdm import tqdm
from .modules import ESNJS


# Experiment parameters
reservoir_size = 100
connectivity = 0.1
spectral_radius = 1.3
leaky_rate = 0.2
batch_size = 10
input_scaling = 0.6
ridge_param = 0.0
bias_scaling = 1.0
image_size = 15
degrees = [30, 60, 60]
n_digits = 10
block_size = 100
input_size = (len(degrees) + 1) * image_size
training_size = 60000
test_size = 10000
use_cuda = False and torch.cuda.is_available()

# MNIST data set train
train_loader = torch.utils.data.DataLoader(
    echotorch.datasets.ImageToTimeseries(
        torchvision.datasets.MNIST(
            root=".",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                echotorch.transforms.images.Concat([
                    echotorch.transforms.images.CropResize(size=image_size),
                    torchvision.transforms.Compose([
                        echotorch.transforms.images.Rotate(degree=degrees[0]),
                        echotorch.transforms.images.CropResize(size=image_size)
                    ]),
                    torchvision.transforms.Compose([
                        echotorch.transforms.images.Rotate(degree=degrees[1]),
                        echotorch.transforms.images.CropResize(size=image_size)
                    ]),
                    torchvision.transforms.Compose([
                        echotorch.transforms.images.Rotate(degree=degrees[2]),
                        echotorch.transforms.images.CropResize(size=image_size)
                    ])
                ],
                    sequential=True
                ),
                torchvision.transforms.ToTensor()
            ]),
            target_transform=echotorch.transforms.targets.ToOneHot(class_size=n_digits)
        ),
        n_images=block_size
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
                    echotorch.transforms.images.CropResize(size=image_size),
                    torchvision.transforms.Compose([
                        echotorch.transforms.images.Rotate(degree=degrees[0]),
                        echotorch.transforms.images.CropResize(size=image_size)
                    ]),
                    torchvision.transforms.Compose([
                        echotorch.transforms.images.Rotate(degree=degrees[1]),
                        echotorch.transforms.images.CropResize(size=image_size)
                    ]),
                    torchvision.transforms.Compose([
                        echotorch.transforms.images.Rotate(degree=degrees[2]),
                        echotorch.transforms.images.CropResize(size=image_size)
                    ])
                ],
                    sequential=True
                ),
                torchvision.transforms.ToTensor()
            ]),
            target_transform=echotorch.transforms.targets.ToOneHot(class_size=n_digits)
        ),
        n_images=block_size
    ),
    batch_size=batch_size,
    shuffle=False
)

# Internal matrix
w_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
    connectivity=connectivity,
    spetral_radius=spectral_radius
)

# Input weights
win_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
    connectivity=connectivity,
    scale=input_scaling,
    apply_spectral_radius=False
)

# Bias vector
wbias_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
    connectivity=connectivity,
    scale=bias_scaling,
    apply_spectral_radius=False
)

# New ESN-JS module
esn = ESNJS(
    input_dim=input_size,
    image_size=image_size,
    hidden_dim=reservoir_size,
    leaky_rate=leaky_rate,
    ridge_param=ridge_param,
    w_generator=w_generator,
    win_generator=win_generator,
    wbias_generator=wbias_generator
)

# Show the model
print(esn)

# Use cuda ?
if use_cuda:
    esn.cuda()
# end if

# For each training sample
with tqdm(total=training_size) as pbar:
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Remove channel
        data = data.reshape(batch_size, 1500, 60)

        # To Variable
        inputs, targets = Variable(data.double()), Variable(targets.double())

        # CUDA
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # end if

        # Feed ESN
        states = esn(inputs, targets)

        # Update bar
        pbar.update(batch_size * block_size)
    # end for
# end with

# Finish training
esn.finalize()

# Total number of right prediction
true_positives = 0.0

# For each test batch
with tqdm(total=test_size) as pbar:
    for batch_idx, (data, targets) in enumerate(test_loader):
        # Remove channel
        data = data.reshape(batch_size, 1500, 60)

        # To Variable
        inputs, targets = Variable(data.double()), Variable(targets.double())

        # CUDA
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # end if

        # Feed ESN
        prediction = esn(inputs, None)

        # Predicted and truth
        _, predicted_class = prediction.max(2)
        _, true_class = targets.max(2)

        # Matching prediction
        true_positives += torch.sum(predicted_class == true_class)

        # Update bar
        pbar.update(batch_size * block_size)
    # end for
# end with

# Show accuracy
print("Error rate : {}".format(100.0 - (true_positives / float(test_size) * 100.0)))
