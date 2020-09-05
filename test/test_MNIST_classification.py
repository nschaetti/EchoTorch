# -*- coding: utf-8 -*-
#
# File : test/test_matrix_generation
# Description : Matrix generation test case.
# Date : 17th of June, 2020
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

import echotorch.utils

import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets
from .modules import ESNJS

from . import EchoTorchTestCase


# Test case : MNIST digit recognition
class Test_MNIST_Classification(EchoTorchTestCase):
    """
    Test case : MNIST digit recognition
    """

    #region PUBLIC
    #endregion PUBLIC

    #region PRIVATE

    # Run the MNIST experiment
    def MNIST_classification(self, expected_error_rate, reservoir_size=100, connectivity=0.1, spectral_radius=1.3,
                             leaky_rate=0.2, batch_size=10, input_scaling=0.6, ridge_param=0.0, bias_scaling=1.0,
                             image_size=15, degrees=[30, 60, 60], minimum_edges=10, block_size=100, places=2):
        """
        Run the MNIST experiment
        :param expected_error_rate:
        :param reservoir_size:
        :param connectivity:
        :param spectral_radius:
        :param leaky_rate:
        :param batch_size:
        :param input_scaling:
        :param ridge_param:
        :param bias_scaling:
        :param image_size:
        :param degrees:
        :param block_size:
        :param places:
        :return:
        """
        # Experiment parameters
        n_digits = 10
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
            spetral_radius=spectral_radius,
            minimum_edges=minimum_edges
        )

        # Input weights
        win_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
            connectivity=connectivity,
            scale=input_scaling,
            apply_spectral_radius=False,
            minimum_edges=minimum_edges
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

        # Use cuda ?
        if use_cuda:
            esn.cuda()
        # end if

        # For each training sample
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Print batch idx
            print(batch_idx)

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
        # end for

        # Finish training
        esn.finalize()

        # Total number of right prediction
        true_positives = 0.0

        # For each test batch
        for batch_idx, (data, targets) in enumerate(test_loader):
            # Print batch idx
            print(batch_idx)

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
            true_positives += torch.sum(predicted_class == true_class).item()
        # end for

        # Test error rate
        self.assertAlmostEqual(1.0 - (true_positives / float(test_size)), expected_error_rate, places)
    # end MNIST_classification

    #endregion PRIVATE

    #region TESTS

    # Test MNIST classification with 10 neurons
    def test_MNIST_classification_10neurons(self):
        """
        Test MNIST classification with 10 neurons
        """
        # Init. random number generator
        echotorch.utils.manual_seed(1)

        # Call experience
        self.MNIST_classification(
            reservoir_size=10,
            expected_error_rate=0.11099999999999999,
            places=2
        )
    # end test_MNIST_classification_10neurons

    # Test MNIST classification with 10 neurons, low connectivity
    def test_MNIST_classification_10neurons_low_connectivity(self):
        """
        Test MNIST classification with 10 neurons, low connectivity
        """
        # Init. random number generator
        echotorch.utils.manual_seed(1)

        # Call experience
        self.MNIST_classification(
            reservoir_size=10,
            connectivity=0.01,
            minimum_edges=20,
            expected_error_rate=0.11870000000000003,
            places=2
        )
    # end test_MNIST_classification_10neurons_low_connectivity

    # Test MNIST classification with 10 neurons
    def test_MNIST_classification_10neurons_alternate_rotations(self):
        """
        Test MNIST classification with 10 neurons
        """
        # Init. random number generator
        echotorch.utils.manual_seed(1)

        # Call experience
        self.MNIST_classification(
            reservoir_size=10,
            degrees=[45, 45, 45],
            expected_error_rate=0.11419999999999997,
            places=2
        )
    # end test_MNIST_classification_10neurons_alternate_rotations

    # Test MNIST classification with 50 neurons
    def test_MNIST_classification_50neurons(self):
        """
        Test MNIST classification with 50 neurons
        """
        # Init. random number generator
        echotorch.utils.manual_seed(1)

        # Call experience
        self.MNIST_classification(
            reservoir_size=50,
            expected_error_rate=0.045499999999999985,
            places=2
        )
    # end test_MNIST_classification_50neurons

    #endregion TESTS

# end Test_MNIST_Classification
