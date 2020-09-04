# -*- coding: utf-8 -*-
#
# File : echotorch/examples/optimization/random_search.py
# Description : Optimize hyperparameters of an ESN with a random search.
# Date : 19 August, 2020
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
import echotorch.nn.reservoir as etrs
import echotorch.utils
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

# Batch size (how many sample processed at the same time?)
batch_size = 1

# Predicted/target plot length
plot_length = 200

# Use CUDA?
use_cuda = False
use_cuda = torch.cuda.is_available() if use_cuda else False


# Function to test the ESN with specific hyperparameters
def evaluation_function(parameters, datasets, n_samples=5):
    """
    Test the ESN with specific parameters on NARMA-10
    :param parameters: Dictionary with parameters values
    :param datasets: The dataset for the evaluation
    :param n_samples: How many samples to test the model ?
    :return: A tuple (model, fitness value)
    """
    # Reservoir hyper-parameters
    spectral_radius = parameters['spectral_radius']
    leaky_rate = parameters['leaky_rate']
    input_dim = 1
    reservoir_size = parameters['reservoir_size']
    connectivity = parameters['connectivity']
    ridge_param = parameters['ridge_param']
    input_scaling = parameters['input_scaling']
    bias_scaling = parameters['bias_scaling']

    # Data loader
    trainloader = DataLoader(datasets[0], batch_size=batch_size, shuffle=False, num_workers=1)
    testloader = DataLoader(datasets[1], batch_size=batch_size, shuffle=False, num_workers=1)

    # Average NRMSE
    NRMSE_average = 0.0

    # For each samples
    for n in range(n_samples):
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

        # Create a Leaky-integrated ESN,
        # with least-square training algo.
        # esn = etrs.ESN(
        esn = etrs.LiESN(
            input_dim=input_dim,
            hidden_dim=reservoir_size,
            output_dim=1,
            leaky_rate=leaky_rate,
            learning_algo='inv',
            w_generator=w_generator,
            win_generator=win_generator,
            wbias_generator=wbias_generator,
            ridge_param=ridge_param
        )

        # Transfer in the GPU if possible
        if use_cuda:
            esn.cuda()
        # end if

        # For each batch
        for data in trainloader:
            # Inputs and outputs
            inputs, targets = data

            # Transform data to Variables
            inputs, targets = Variable(inputs), Variable(targets)
            if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()

            # ESN need inputs and targets
            esn(inputs, targets)
        # end for

        # Now we finalize the training by
        # computing the output matrix Wout.
        esn.finalize()

        # Get the first sample in test set,
        # and transform it to Variable.
        dataiter = iter(testloader)
        test_u, test_y = dataiter.next()
        test_u, test_y = Variable(test_u), Variable(test_y)
        if use_cuda: test_u, test_y = test_u.cuda(), test_y.cuda()

        # Make a prediction with our trained ESN
        y_predicted = esn(test_u)

        # Add to sum of NRMSE
        NRMSE_average += echotorch.utils.nrmse(y_predicted.data, test_y.data)
    # end for

    # Print test MSE and NRMSE
    return esn, NRMSE_average / n_samples
# end evaluation_function
