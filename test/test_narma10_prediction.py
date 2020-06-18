# -*- coding: utf-8 -*-
#
# File : test/narma10_prediction
# Description : NARMA-10 prediction test case.
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
import unittest
from unittest import TestCase

from echotorch.datasets.NARMADataset import NARMADataset
import echotorch.nn.reservoir as etrs
import echotorch.utils.matrix_generation as mg
import echotorch.utils

import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

import numpy as np

from .EchoTorchTestCase import EchoTorchTestCase


# Test case : NARMA10 timeseries prediction.
class Test_NARMA10_Prediction(EchoTorchTestCase):
    """
    Test NARMA10 timeseries prediction
    """

    #region PUBLIC

    #endregion PUBLIC

    #region TESTS

    # Test NARMA-10 prediction with default hyper-parameters (Nx=100, SP=0.99)
    def test_narma10_prediction_esn(self):
        """
        Test NARMA-10 prediction with default hyper-parameters (Nx=100, SP=0.99)
        """
        # Run NARMA-10 prediction with default hyper-parameters
        train_mse, train_nrmse, test_mse, test_nrmse, y_train_pred, y_test_pred, w, w_in, w_bias = self.narma10_prediction()
        print(w)
        print(w_in)
        print(w_bias)
        # Check results
        self.assertAlmostEqual(train_mse, 0.0014486080035567284, places=3)
        self.assertAlmostEqual(train_nrmse, 0.3550744227519557, places=2)
        self.assertAlmostEqual(test_mse, 0.0020347298122942448, places=3)
        self.assertAlmostEqual(test_nrmse, 0.37887486593306385, places=2)
    # end test_narma10_prediction

    # Test NARMA-10 prediction with 500 neurons
    def test_narma10_prediction_esn_500neurons(self):
        """
        Test NARMA-10 prediction with 500 neurons
        """
        # Run NARMA-10 prediction with default hyper-parameters
        train_mse, train_nrmse, test_mse, test_nrmse, y_train_pred, y_test_pred = self.narma10_prediction(
            reservoir_size=500
        )

        # Check results
        self.assertAlmostEqual(train_mse, 0.0489947535097599, places=2)
        self.assertAlmostEqual(train_nrmse, 2.0649937667937337, places=1)
        self.assertAlmostEqual(test_mse, 0.05040527135133743, places=2)
        self.assertAlmostEqual(test_nrmse, 1.885733842954738, places=1)
    # end test_narma10_prediction_500neurons

    # Test NARMA-10 prediction with leaky-rate 0.5 (Nx=100, SP=0.99, LR=0.5)
    def test_narma10_prediction_liesn(self):
        """
        Test NARMA-10 prediction with leaky-rate 0.5 (Nx=100, SP=0.99, LR=0.5)
        """
        # Run NARMA-10 prediction with default hyper-parameters
        train_mse, train_nrmse, test_mse, test_nrmse, y_train_pred, y_test_pred = self.narma10_prediction(leaky_rate=0.5)

        # Check results
        self.assertAlmostEqual(train_mse, 0.036606427282094955, places=3)
        self.assertAlmostEqual(train_nrmse, 1.7849359355510792, places=2)
        self.assertAlmostEqual(test_mse, 0.038768090307712555, places=3)
        self.assertAlmostEqual(test_nrmse, 1.6537871853546815, places=2)
    # end test_narma10_prediction

    #endregion TESTS

    #region PRIVATE

    # Run NARMA-10 prediction with classic ESN
    def narma10_prediction(self, train_sample_length=5000, test_sample_length=1000, n_train_samples=1, n_test_samples=1,
                           batch_size=1, reservoir_size=100, leaky_rate=1.0, spectral_radius=0.99, connectivity=0.1,
                           input_scaling=1.0, bias_scaling=0.0, ridge_param=0.0000001):
        """
        Run NARMA-10 prediction with classic ESN
        :param train_sample_length: Training sample length
        :param test_sample_length: Test sample length
        :param n_train_samples: Number of training samples
        :param n_test_samples: Number of test samples
        :param batch_size: Batch-size
        :param reservoir_size: Reservoir size (how many units in the reservoir)
        :param leaky_rate: Leary rate
        :param spectral_radius: Spectral radius
        :param connectivity: ratio of zero in internal weight matrix W
        :param input_scaling: Input scaling
        :param bias_scaling: Bias scaling
        :param ridge_param: Ridge parameter (regularization)
        :return: train MSE, train NRMSE, test MSE, test NRMSE
        """
        # Use CUDA?
        use_cuda = False
        use_cuda = torch.cuda.is_available() if use_cuda else False

        # Manual seed initialisation
        np.random.seed(1)
        torch.manual_seed(1)

        # NARMA30 dataset
        narma10_train_dataset = NARMADataset(train_sample_length, n_train_samples, system_order=10)
        narma10_test_dataset = NARMADataset(test_sample_length, n_test_samples, system_order=10)

        # Data loader
        trainloader = DataLoader(narma10_train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        testloader = DataLoader(narma10_test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        # Matrix generator for W
        w_matrix_generator = mg.matrix_factory.get_generator(
            name='normal',
            connectivity=connectivity,
            spectral_radius=spectral_radius
        )

        # Matrix generator for Win
        win_matrix_generator = mg.matrix_factory.get_generator(
            name='normal',
            connectivity=connectivity,
            scale=input_scaling,
            apply_spectral_radius=False
        )

        # Matrix generator for Wbias
        wbias_matrix_generator = mg.matrix_factory.get_generator(
            name='normal',
            connectivity=connectivity,
            scale=bias_scaling,
            apply_spectral_radius=False
        )

        # Create a Leaky-integrated ESN,
        # with least-square training algo.
        esn = etrs.LiESN(
            input_dim=1,
            hidden_dim=reservoir_size,
            output_dim=1,
            spectral_radius=spectral_radius,
            leaky_rate=leaky_rate,
            learning_algo='inv',
            w_generator=w_matrix_generator,
            win_generator=win_matrix_generator,
            wbias_generator=wbias_matrix_generator,
            input_scaling=input_scaling,
            bias_scaling=bias_scaling,
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

        # Get the first sample in training set,
        # and transform it to Variable.
        dataiter = iter(trainloader)
        train_u, train_y = dataiter.next()
        train_u, train_y = Variable(train_u), Variable(train_y)
        if use_cuda: train_u, train_y = train_u.cuda(), train_y.cuda()

        # Make a prediction with our trained ESN
        y_train_predicted = esn(train_u)

        # Get the first sample in test set,
        # and transform it to Variable.
        dataiter = iter(testloader)
        test_u, test_y = dataiter.next()
        test_u, test_y = Variable(test_u), Variable(test_y)
        if use_cuda: test_u, test_y = test_u.cuda(), test_y.cuda()

        # Make a prediction with our trained ESN
        y_test_predicted = esn(test_u)

        return (
            echotorch.utils.mse(y_train_predicted.data, train_y.data),
            echotorch.utils.nrmse(y_train_predicted.data, train_y.data),
            echotorch.utils.mse(y_test_predicted.data, test_y.data),
            echotorch.utils.nrmse(y_test_predicted.data, test_y.data),
            y_train_predicted,
            y_test_predicted,
            esn.cell.w,
            esn.cell.w_in,
            esn.cell.w_bias
        )
    # end narma10_prediction

    #endregion PRIVATE

# end test_narma10_prediction


# Run test
if __name__ == '__main__':
    unittest.main()
# end if
