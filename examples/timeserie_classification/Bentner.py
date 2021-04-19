# -*- coding: utf-8 -*-
#
# File : examples/timeserie_prediction/narma10_esn
# Description : NARMA-10 prediction with ESN.
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
import sys
import torch
import echotorch.nn.reservoir as etrs
import echotorch.utils
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    # A pytorch dataset class for holding data for a text classification task.
    def __init__(
        self,
        sample_len,
        sample_offset=0,
        transform=None,
    ):

        self.sample_len = sample_len

        xy = np.loadtxt(
            "DD_HTP_AeroCoefficients_AoA10.dat",
            delimiter="    ",
            usecols=(0, 3),
            dtype=np.float32,
        )
        # print("xy: {}".format(xy))
        self.n_samples = sample_len
        # self.n_samples = xy.shape[0]
        self.x = xy[sample_offset : sample_offset + sample_len + 1, [0]]
        self.y = xy[sample_offset : sample_offset + sample_len + 1, [1]]
        # print("#############")
        # print(self.x)
        # print("y: {}".format(self.y))
        self.transform = transform
    # end __init__

    def __getitem__(self, index):  # self.x[index],
        if index > 11500:
            print(index)
        sample = self.y[:-1], self.y[1:]

        if self.transform:
            sample = self.transform(sample)

        return sample
    # end __getitem__

    def __len__(self):
        return self.n_samples
    # end __len__

# end CustomDataset


class ToTensor:

    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    # end __call__

# end ToTensor


# Length of training samples
train_sample_length = 11500

# Length of test samples
test_sample_length = 4000

# How many training/test samples
n_train_samples = 1
n_test_samples = 1

# Batch size (how many sample processed at the same time?)
batch_size = 1

# Reservoir hyper-parameters
spectral_radius = 1.2
leaky_rate = 1
input_dim = 1
reservoir_size = 30
connectivity = 0.1954
ridge_param = 0.00000409
# ridge_param = 0.0001
input_scaling = 0.9252
bias_scaling = 0.079079

# Predicted/target plot length
plot_length = 15500

# Use CUDA?
use_cuda = False
use_cuda = torch.cuda.is_available() if use_cuda else False

# Manual seed initialisation
np.random.seed(1)
torch.manual_seed(1)

# Dataset
train_dataset = CustomDataset(train_sample_length, transform=ToTensor())
test_dataset = CustomDataset(test_sample_length, sample_offset=train_sample_length, transform=ToTensor())

# print(train_dataset)

# Data loader
trainloader = DataLoader(
    train_dataset, batch_size=15500, shuffle=False, num_workers=0
)
testloader = DataLoader(
    test_dataset, batch_size=15500, shuffle=False, num_workers=0
)

# Internal matrix
w_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
    connectivity=connectivity, spetral_radius=spectral_radius
)

# Input weights
win_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
    connectivity=connectivity, scale=input_scaling, apply_spectral_radius=False
)

# Bias vector
wbias_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
    connectivity=connectivity, scale=bias_scaling, apply_spectral_radius=False
)

# Create a Leaky-integrated ESN,
# with least-square training algo.
# esn = etrs.ESN(
esn = etrs.LiESN(
    input_dim=input_dim,
    hidden_dim=reservoir_size,
    output_dim=1,
    leaky_rate=leaky_rate,
    learning_algo="inv",
    w_generator=w_generator,
    win_generator=win_generator,
    wbias_generator=wbias_generator,
    ridge_param=ridge_param,
)

# Transfer in the GPU if possible
if use_cuda:
    esn.cuda()
# end if

# For each batch
for data in trainloader:
    # Inputs and outputs
    # print("??????????")
    # print(data)
    inputs, targets = data

    # Transform data to Variables
    # inputs, targets = Variable(inputs), Variable(targets)
    inputs, targets = (
        Variable(inputs).resize_(1, train_sample_length, 1),
        Variable(targets).resize_(1, train_sample_length, 1),
    )
    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    # print("%%%%%%%")
    # print(inputs)
    # print(targets)
    # ESN need inputs and targets
    esn(inputs, targets)
# end for

# Now we finalize the training by
# computing the output matrix Wout.
esn.finalize()

# print(esn.w_out.size())
# print(esn.w_out)

# Get the first sample in training set,
# and transform it to Variable.
dataiter = iter(trainloader)
train_u, train_y = dataiter.next()
train_u, train_y = Variable(train_u).resize_(1, train_sample_length, 1), Variable(train_y).resize_(
    1, train_sample_length, 1
)
if use_cuda:
    train_u, train_y = train_u.cuda(), train_y.cuda()

# Make a prediction with our trained ESN
y_predicted = esn(train_u)  # vorhersage

# Print training MSE and NRMSE
print(u"Train MSE: {}".format(echotorch.utils.mse(y_predicted.data, train_y.data)))
print(u"Train NRMSE: {}".format(echotorch.utils.nrmse(y_predicted.data, train_y.data)))
print(u"")

# Get the first sample in test set,
# and transform it to Variable.
dataiter = iter(testloader)
test_u, test_y = dataiter.next()
test_u, test_y = (
    Variable(test_u).resize_(1, test_sample_length, 1),
    Variable(test_y).resize_(1, test_sample_length, 1),
)
if use_cuda:
    test_u, test_y = test_u.cuda(), test_y.cuda()

# Make a prediction with our trained ESN
y_predicted = esn(test_u)
# print(y_predicted.reshape((1, test_sample_length)))

# Print test MSE and NRMSE
print(u"Test MSE: {}".format(echotorch.utils.mse(y_predicted.data, test_y.data)))
print(u"Test NRMSE: {}".format(echotorch.utils.nrmse(y_predicted.data, test_y.data)))
print(u"")

# Show target and predicted
# xaxis= np.loadtxt('DD_HTP_AeroCoefficients_AoA10.dat', usecols=0)
full_dataset = CustomDataset(
    test_sample_length, sample_offset=train_sample_length, transform=ToTensor()
)
# plt.plot(xaxis[train_sample_length:train_sample_length + test_sample_length], test_y[0, train_sample_length:train_sample_length + test_sample_length, 0].data, 'r')
plt.figure(figsize=(10, 8))
plt.plot(full_dataset.x, full_dataset.y, "r")
plt.plot(test_dataset.x, torch.squeeze(y_predicted), "b")
plt.show()
