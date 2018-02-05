# -*- coding: utf-8 -*-
#
# File : examples/timeserie_prediction/switch_attractor_esn
# Description : NARMA 30 prediction with ESN.
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
from echotorch.datasets.SwitchAttractorDataset import SwitchAttractorDataset
import echotorch.nn as etnn
import echotorch.utils
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import numpy as np
import mdp
import matplotlib.pyplot as plt

# Dataset params
train_sample_length = 1000
test_sample_length = 1000
n_train_samples = 40
n_test_samples = 10
batch_size = 1
spectral_radius = 0.9
leaky_rate = 1.0
input_dim = 1
n_hidden = 100

# Use CUDA?
use_cuda = False
use_cuda = torch.cuda.is_available() if use_cuda else False

# Manual seed
mdp.numx.random.seed(1)
np.random.seed(2)
torch.manual_seed(1)

# Switch attractor dataset
switch_train_dataset = SwitchAttractorDataset(train_sample_length, n_train_samples, seed=1)
switch_test_dataset = SwitchAttractorDataset(test_sample_length, n_test_samples, seed=10)

# Data loader
trainloader = DataLoader(switch_train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = DataLoader(switch_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ESN cell
esn = etnn.LiESN(input_dim=input_dim, hidden_dim=n_hidden, output_dim=1, spectral_radius=spectral_radius,
                 learning_algo='inv', leaky_rate=leaky_rate, feedbacks=True)
if use_cuda:
    esn.cuda()
# end if

# For each batch
for data in trainloader:
    # Inputs and outputs
    inputs, targets = data

    # To variable
    inputs, targets = Variable(inputs), Variable(targets)
    if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
    # plt.plot(targets.data[0].numpy(), c='b')
    # plt.plot(y_predicted.data[0, :, 0].numpy(), c='r')
    # plt.show()
    # Accumulate xTx and xTy
    esn(inputs, targets)
# end for

# Finalize training
esn.finalize()

# For each batch
for data in testloader:
    # Test MSE
    test_u, test_y = data
    test_u, test_y = Variable(test_u), Variable(test_y)
    if use_cuda: test_u, test_y = test_u.cuda(), test_y.cuda()
    y_predicted = esn(test_u)
    plt.ylim(ymax=10)
    plt.plot(test_y.data[0].numpy(), c='b')
    plt.plot(y_predicted.data[0, :, 0].numpy(), c='r')
    plt.show()
# end for
