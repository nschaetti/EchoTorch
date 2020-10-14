# -*- coding: utf-8 -*-
#
# File : test/test_hyper-parameters_optimization.py
# Description : Hyper-parameters optimization test cases
# Date : 20th of August, 2020
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

from echotorch.datasets.NARMADataset import NARMADataset
import echotorch.utils.optimization as optim
import torch
import random
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import echotorch.nn.reservoir as etrs
import echotorch.utils
import numpy as np
from . import EchoTorchTestCase


# Test cases : Hyper-parameters optimization
class Test_Hyperparameters_Optimization(EchoTorchTestCase):
    """
    Hyper-parameters optimization
    """

    # region PRIVATE

    # Function to test the ESN on the NARMA-10 task
    def _evaluation_NARMA10(self, parameters, datasets, n_samples=5):
        """
        Test the ESN with specific parameters on NARMA-10
        :param parameters: Dictionary with parameters values
        :param datasets: The dataset for the evaluation
        :param n_samples: How many samples to test the model ?
        :return: A tuple (model, fitness value)
        """
        # Batch size (how many sample processed at the same time?)
        batch_size = 1

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

            # For each batch
            for data in trainloader:
                # Inputs and outputs
                inputs, targets = data

                # Transform data to Variables
                inputs, targets = Variable(inputs), Variable(targets)

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

            # Make a prediction with our trained ESN
            y_predicted = esn(test_u)

            # Add to sum of NRMSE
            NRMSE_average += echotorch.utils.nrmse(y_predicted.data, test_y.data)
        # end for

        # Print test MSE and NRMSE
        return esn, NRMSE_average / n_samples
    # end evaluation_NARMA10

    # endregion PRIVATE

    # region TESTS

    # Test genetic optimization on NARMA-10
    def test_genetic_optimization_NARMA10(self):
        """
        Test genetic optimization on NARMA-10
        """
        # Debug ?
        debug = False

        # Length of training samples
        train_sample_length = 5000

        # Length of test samples
        test_sample_length = 1000

        # How many training/test samples
        n_train_samples = 1
        n_test_samples = 1

        # Manual seed initialisation
        echotorch.utils.random.manual_seed(1)

        # Get a random optimizer
        random_optimizer = optim.optimizer_factory.get_optimizer(
            'genetic',
            iterations=2
        )

        # NARMA10 dataset
        narma10_train_dataset = NARMADataset(train_sample_length, n_train_samples, system_order=10)
        narma10_test_dataset = NARMADataset(test_sample_length, n_test_samples, system_order=10)

        # Parameters ranges
        param_ranges = dict()
        param_ranges['spectral_radius'] = np.linspace(0, 2.0, 1000)
        param_ranges['leaky_rate'] = np.linspace(0.1, 1.0, 1000)
        param_ranges['reservoir_size'] = np.arange(50, 510, 10)
        param_ranges['connectivity'] = np.linspace(0.1, 1.0, 1000)
        param_ranges['ridge_param'] = np.logspace(-10, 2, base=10, num=1000)
        param_ranges['input_scaling'] = np.linspace(0.1, 1.0, 1000)
        param_ranges['bias_scaling'] = np.linspace(0.0, 1.0, 1000)

        # Launch the optimization of hyper-parameter
        _, best_param, best_NRMSE = random_optimizer.optimize(
            self._evaluation_NARMA10,
            param_ranges,
            (narma10_train_dataset, narma10_test_dataset),
            n_samples=5
        )

        # Show the result
        if debug:
            print("Best hyper-parameters found : {}".format(best_param))
            print("Best NRMSE : {}".format(best_NRMSE))
        # end if

        # Test the NRMSE found with optimization
        self.assertLessEqual(
            best_NRMSE,
            0.5,
            msg="NRMSE to high for genetic optimisation, check the implementation!"
        )
    # end test_genetic_optimization_NARMA10

    # Test grid search optimization on NARMA10
    def test_grid_search_optimization_NARMA10(self):
        """
        Test grid search optimization on NARMA10
        """
        # Debug?
        debug = False

        # Length of training samples
        train_sample_length = 5000

        # Length of test samples
        test_sample_length = 1000

        # How many training/test samples
        n_train_samples = 1
        n_test_samples = 1

        # Manual seed initialisation
        echotorch.utils.random.manual_seed(1)

        # Get a random optimizer
        random_optimizer = optim.optimizer_factory.get_optimizer('grid-search')

        # NARMA10 dataset
        narma10_train_dataset = NARMADataset(train_sample_length, n_train_samples, system_order=10)
        narma10_test_dataset = NARMADataset(test_sample_length, n_test_samples, system_order=10)

        # Parameters ranges
        param_ranges = dict()
        param_ranges['spectral_radius'] = np.arange(0, 1.1, 0.5)
        param_ranges['leaky_rate'] = np.arange(0.1, 1.1, 0.5)
        param_ranges['reservoir_size'] = np.arange(100, 510, 200)
        param_ranges['connectivity'] = np.arange(0.1, 1.0, 0.4)
        param_ranges['ridge_param'] = np.logspace(-10, 2, base=10, num=2)
        param_ranges['input_scaling'] = np.arange(0.1, 1.1, 0.4)
        param_ranges['bias_scaling'] = np.arange(0.0, 1.1, 0.5)

        # Launch the optimization of hyper-parameters
        _, best_param, best_NRMSE = random_optimizer.optimize(
            self._evaluation_NARMA10,
            param_ranges,
            (narma10_train_dataset, narma10_test_dataset),
            n_samples=1
        )

        # Show the result
        if debug:
            print("Best hyper-parameters found : {}".format(best_param))
            print("Best NRMSE : {}".format(best_NRMSE))
        # end if

        # Test the NRMSE of the ESN found with optimization
        # self.assertAlmostEqual(best_NRMSE, 1.553938488748105, places=2)
        self.assertLessEqual(
            best_NRMSE,
            1.6,
            msg="NRMSE to high for grid optimisation, check the implementation!"
        )
    # end test_grid_search_optimization_NARMA10

    # Test random optimization on NARMA10
    def test_random_optimization_NARMA10(self):
        """
        Test random optimization on NARMA10
        """
        # Debug?
        debug = False

        # Length of training samples
        train_sample_length = 5000

        # Length of test samples
        test_sample_length = 1000

        # How many training/test samples
        n_train_samples = 1
        n_test_samples = 1

        # Manual seed initialisation
        echotorch.utils.random.manual_seed(1)

        # Get a random optimizer
        random_optimizer = optim.optimizer_factory.get_optimizer('random', R=50)

        # NARMA10 dataset
        narma10_train_dataset = NARMADataset(train_sample_length, n_train_samples, system_order=10)
        narma10_test_dataset = NARMADataset(test_sample_length, n_test_samples, system_order=10)

        # Parameters ranges
        param_ranges = dict()
        param_ranges['spectral_radius'] = np.arange(0, 1.1, 0.1)
        param_ranges['leaky_rate'] = np.arange(0.1, 1.1, 0.1)
        param_ranges['reservoir_size'] = np.arange(50, 500, 50)
        param_ranges['connectivity'] = np.arange(0.1, 1.0, 0.1)
        param_ranges['ridge_param'] = np.logspace(-10, 2, base=10, num=10)
        param_ranges['input_scaling'] = np.arange(0.1, 1.1, 0.1)
        param_ranges['bias_scaling'] = np.arange(0.0, 1.1, 0.1)

        # Launch the optimization of hyper-parameters
        _, best_param, best_NRMSE = random_optimizer.optimize(
            self._evaluation_NARMA10,
            param_ranges,
            (narma10_train_dataset, narma10_test_dataset),
            n_samples=5
        )

        # Show the result
        if debug:
            print("Best hyper-parameters found : {}".format(best_param))
            print("Best NRMSE : {}".format(best_NRMSE))
        # end if

        # Test the NRMSE of the ESN found with optimization
        # self.assertAlmostEqual(best_NRMSE, 0.49092315487463206, places=1)
        self.assertLessEqual(
            best_NRMSE,
            0.5,
            msg="NRMSE to high for random optimisation, check the implementation!"
        )
    # end test_random_optimization_NARMA10

    # endregion TESTS

# end Test_Hyperparameters_Optimization
