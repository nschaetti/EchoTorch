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
        # Run NARMA-10 prediction with default hyper-parameters (64 and 32)
        train_mse, train_nrmse, test_mse, test_nrmse = self.narma10_prediction()
        train_mse32, train_nrmse32, test_mse32, test_nrmse32 = self.narma10_prediction(dtype=torch.float32)

        # Check results for 64 bits
        self.assertAlmostEqual(train_mse, 0.0014486080035567284, places=3)
        self.assertAlmostEqual(train_nrmse, 0.39079351181912997, places=1)
        self.assertAlmostEqual(test_mse, 0.00269576234138083, places=3)
        self.assertAlmostEqual(test_nrmse, 0.43609712777215, places=1)

        # Check results for 32 bits
        self.assertAlmostEqual(train_mse32, 0.0014486080035567284, places=3)
        self.assertAlmostEqual(train_nrmse32, 0.3550744227519557, places=1)
        self.assertAlmostEqual(test_mse32, 0.0020347298122942448, places=3)
        self.assertAlmostEqual(test_nrmse32, 0.37887486593306385, places=1)
    # end test_narma10_prediction

    # Test NARMA-10 prediction with ridge param to 0.1 (Nx=100, SP=0.99)
    def test_narma10_prediction_esn_ridge01(self):
        """
        Test NARMA-10 prediction with default hyper-parameters (Nx=100, SP=0.99)
        """
        # Run NARMA-10 prediction with default hyper-parameters (64 and 32)
        train_mse, train_nrmse, test_mse, test_nrmse = self.narma10_prediction(ridge_param=0.1)
        train_mse32, train_nrmse32, test_mse32, test_nrmse32 = self.narma10_prediction(
            ridge_param=0.1,
            dtype=torch.float32
        )

        # Check results for 64 bits
        self.assertAlmostEqual(train_mse, 0.0055206307312779, places=3)
        self.assertAlmostEqual(train_nrmse, 0.6931676774014105, places=1)
        self.assertAlmostEqual(test_mse, 0.007794506000496885, places=3)
        self.assertAlmostEqual(test_nrmse, 0.7415436548404447, places=1)

        # Check results for 32 bits
        self.assertAlmostEqual(train_mse32, 0.005962355528026819, places=3)
        self.assertAlmostEqual(train_nrmse32, 0.7203654917258767, places=1)
        self.assertAlmostEqual(test_mse32, 0.008289368823170662, places=3)
        self.assertAlmostEqual(test_nrmse32, 0.76472123969956, places=1)
    # end test_narma10_prediction_esn_ridge01

    # Test NARMA-10 prediction with ridge param to 0.001 (Nx=100, SP=0.99)
    def test_narma10_prediction_esn_ridge001(self):
        """
        Test NARMA-10 prediction with default hyper-parameters (Nx=100, SP=0.99)
        """
        # Run NARMA-10 prediction with default hyper-parameters (64 and 32)
        train_mse, train_nrmse, test_mse, test_nrmse = self.narma10_prediction(ridge_param=0.01)
        train_mse32, train_nrmse32, test_mse32, test_nrmse32 = self.narma10_prediction(
            ridge_param=0.01,
            dtype=torch.float32
        )

        # Check results for 64 bits
        self.assertAlmostEqual(train_mse, 0.0037603671659062763, places=3)
        self.assertAlmostEqual(train_nrmse, 0.5720830492793262, places=1)
        self.assertAlmostEqual(test_mse, 0.005600038439979167, places=3)
        self.assertAlmostEqual(test_nrmse, 0.6285472481319053, places=1)

        # Check results for 32 bits
        self.assertAlmostEqual(train_mse32, 0.003447971772402525, places=3)
        self.assertAlmostEqual(train_nrmse32, 0.5478047777622833, places=1)
        self.assertAlmostEqual(test_mse32, 0.005058450624346733, places=3)
        self.assertAlmostEqual(test_nrmse32, 0.5973806805561077, places=1)
    # end test_narma10_prediction_esn_ridge001

    # Test NARMA-10 prediction with ridge param to 10 (Nx=100, SP=0.99)
    def test_narma10_prediction_esn_ridge10(self):
        """
        Test NARMA-10 prediction with default hyper-parameters (Nx=100, SP=0.99)
        """
        # Run NARMA-10 prediction with default hyper-parameters (64 and 32)
        train_mse, train_nrmse, test_mse, test_nrmse = self.narma10_prediction(ridge_param=10)
        train_mse32, train_nrmse32, test_mse32, test_nrmse32 = self.narma10_prediction(
            ridge_param=10.0,
            dtype=torch.float32
        )

        # Check results for 64 bits
        self.assertAlmostEqual(train_mse, 0.05097953801874421, places=2)
        self.assertAlmostEqual(train_nrmse, 2.106405119788818, places=1)
        self.assertAlmostEqual(test_mse, 0.062314379492128924, places=2)
        self.assertAlmostEqual(test_nrmse, 2.0967012913437437, places=1)

        # Check results for 32 bits
        self.assertAlmostEqual(train_mse32, 0.0878116637468338, places=2)
        self.assertAlmostEqual(train_nrmse32, 2.7645220092625897, places=1)
        self.assertAlmostEqual(test_mse32, 0.10268372297286987, places=1)
        self.assertAlmostEqual(test_nrmse32, 2.691492796282878, places=1)
    # end test_narma10_prediction_esn_ridge10

    # Test NARMA-10 prediction with 500 neurons
    def test_narma10_prediction_esn_500neurons(self):
        """
        Test NARMA-10 prediction with 500 neurons
        """
        # Run NARMA-10 prediction with default hyper-parameters (64 and 32 bits)
        train_mse, train_nrmse, test_mse, test_nrmse = self.narma10_prediction(reservoir_size=500)
        train_mse32, train_nrmse32, test_mse32, test_nrmse32 = self.narma10_prediction(
            reservoir_size=500,
            dtype=torch.float32
        )

        # Check results for 64 bits
        self.assertAlmostEqual(train_mse, 0.00040123554059248175, places=2)
        self.assertAlmostEqual(train_nrmse, 0.18687174945460525, places=1)
        self.assertAlmostEqual(test_mse, 0.000787773357517867, places=2)
        self.assertAlmostEqual(test_nrmse, 0.2357453144653597, places=1)

        # Check results for 32 bits
        self.assertAlmostEqual(train_mse32, 0.0489947535097599, places=1)
        self.assertLessEqual(train_nrmse32, 2.1)
        self.assertAlmostEqual(test_mse32, 0.05040527135133743, places=1)
        self.assertLessEqual(test_nrmse32, 2.0)
    # end test_narma10_prediction_500neurons

    # Test NARMA-10 prediction with leaky-rate 0.5 (Nx=100, SP=0.99, LR=0.5)
    def test_narma10_prediction_liesn(self):
        """
        Test NARMA-10 prediction with leaky-rate 0.5 (Nx=100, SP=0.99, LR=0.5)
        """
        # Run NARMA-10 prediction with default hyper-parameters (32 and 64 bits)
        train_mse, train_nrmse, test_mse, test_nrmse = self.narma10_prediction(leaky_rate=0.5)
        train_mse32, train_nrmse32, test_mse32, test_nrmse32 = self.narma10_prediction(
            leaky_rate=0.5,
            dtype=torch.float32
        )

        # Check results
        self.assertAlmostEqual(train_mse, 0.0024004888199628395, places=3)
        self.assertAlmostEqual(train_nrmse, 0.45708166187229005, places=2)
        self.assertAlmostEqual(test_mse, 0.0032115068445433756, places=3)
        self.assertAlmostEqual(test_nrmse, 0.4759889286000261, places=2)

        # Check results
        self.assertAlmostEqual(train_mse32, 0.036606427282094955, places=1)
        self.assertLessEqual(train_nrmse32, 1.8)
        self.assertAlmostEqual(test_mse32, 0.038768090307712555, places=1)
        self.assertLessEqual(test_nrmse32, 1.8)
    # end test_narma10_prediction

    #endregion TESTS

    #region PRIVATE

    # Run NARMA-10 prediction with classic ESN
    def narma10_prediction(self, train_sample_length=5000, test_sample_length=1000, n_train_samples=1, n_test_samples=1,
                           batch_size=1, reservoir_size=100, leaky_rate=1.0, spectral_radius=0.99, connectivity=0.1,
                           input_scaling=1.0, bias_scaling=0.0, ridge_param=0.0000001, dtype=torch.float64):
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
        echotorch.utils.manual_seed(1)

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
            spectral_radius=spectral_radius,
            dtype=dtype
        )

        # Matrix generator for Win
        win_matrix_generator = mg.matrix_factory.get_generator(
            name='normal',
            connectivity=connectivity,
            scale=input_scaling,
            apply_spectral_radius=False,
            dtype=dtype
        )

        # Matrix generator for Wbias
        wbias_matrix_generator = mg.matrix_factory.get_generator(
            name='normal',
            connectivity=connectivity,
            scale=bias_scaling,
            apply_spectral_radius=False,
            dtype=dtype
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
            ridge_param=ridge_param,
            dtype=dtype
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
            if dtype == torch.float64: inputs, targets = inputs.double(), targets.double()
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
        if dtype == torch.float64: train_u, train_y = train_u.double(), train_y.double()
        train_u, train_y = Variable(train_u), Variable(train_y)
        if use_cuda: train_u, train_y = train_u.cuda(), train_y.cuda()

        # Make a prediction with our trained ESN
        y_train_predicted = esn(train_u)

        # Get the first sample in test set,
        # and transform it to Variable.
        dataiter = iter(testloader)
        test_u, test_y = dataiter.next()
        if dtype == torch.float64: test_u, test_y = test_u.double(), test_y.double()
        test_u, test_y = Variable(test_u), Variable(test_y)
        if use_cuda: test_u, test_y = test_u.cuda(), test_y.cuda()

        # Make a prediction with our trained ESN
        y_test_predicted = esn(test_u)

        return (
            echotorch.utils.mse(y_train_predicted.data, train_y.data),
            echotorch.utils.nrmse(y_train_predicted.data, train_y.data),
            echotorch.utils.mse(y_test_predicted.data, test_y.data),
            echotorch.utils.nrmse(y_test_predicted.data, test_y.data)
        )
    # end narma10_prediction

    #endregion PRIVATE

# end test_narma10_prediction


# Run test
if __name__ == '__main__':
    unittest.main()
# end if
