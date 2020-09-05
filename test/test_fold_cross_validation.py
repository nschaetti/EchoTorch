# -*- coding: utf-8 -*-
#
# File : test/test_fold_cross_validation.py
# Description : Test fold cross validation for evaluation.
# Date : 5th of September, 2020
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
import os
import echotorch.utils
from .EchoTorchTestCase import EchoTorchTestCase
import numpy as np
import torch
import echotorch.nn.conceptors as ecnc
import echotorch.utils.matrix_generation as mg
import echotorch.utils
import echotorch.datasets as etds
import echotorch.evaluation as val
from echotorch.datasets import DatasetComposer
from echotorch.nn.Node import Node
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from .modules import DummyDataset


# Test case : fold cross validation
class Test_Fold_Cross_Validation(EchoTorchTestCase):
    """
    Test fold cross validation
    """
    # region BODY

    # region PUBLIC

    # Test fold cross validation
    def fold_cross_validation(self, sample_len, input_dim, n_classes, n_samples, n_folds, with_dev, shuffle,
                              sample_indices, train_size, shuffle_cv):
        """
        Test fold cross validation
        :param sample_len: Sample length
        :param input_dim: Timeseries input dimension
        :param n_classes: Number of classes
        :param n_samples: Number of samples
        :param n_folds: Number of folds
        :param with_dev: Test with dev
        :param shuffle: Shuffle dataset before testing
        :param sample_indices: Indices of samples used for the cross validation.
        :param train_size: Training set size (0.0 - 1.0)
        :param shuffle_cv: Shuffle dataset before the cross validation.
        """
        # Dummy dataset
        dummy_dataset = DummyDataset(
            sample_len=sample_len,
            n_samples=n_samples,
            input_dim=input_dim,
            n_classes=n_classes
        )

        # Validation set ratio
        if with_dev:
            dev_ratio = 0.5
        else:
            dev_ratio = 0.0
        # end if

        # Cross validation training
        dummy_cv_train_dataset = val.CrossValidationWithDev(
            root_dataset=dummy_dataset, k=n_folds, mode='train', samples_indices=sample_indices, train_size=train_size,
            dev_ratio=dev_ratio, shuffle=shuffle_cv
        )

        # Cross validation set
        dummy_cv_dev_dataset = val.CrossValidationWithDev(
            root_dataset=dummy_dataset, k=n_folds, mode='dev', samples_indices=sample_indices, train_size=train_size,
            dev_ratio=dev_ratio, shuffle=shuffle_cv
        )

        # Cross test set
        dummy_cv_test_dataset = val.CrossValidationWithDev(
            root_dataset=dummy_dataset, k=n_folds, mode='test', samples_indices=sample_indices, train_size=train_size,
            dev_ratio=dev_ratio, shuffle=shuffle_cv
        )

        # Dataset loaders
        dummy_train_dataset_loader = DataLoader(dataset=dummy_cv_train_dataset, batch_size=1, shuffle=shuffle)
        dummy_dev_dataset_loader = DataLoader(dataset=dummy_cv_dev_dataset, batch_size=1, shuffle=shuffle)
        dummy_test_dataset_loader = DataLoader(dataset=dummy_cv_test_dataset, batch_size=1, shuffle=shuffle)

        # For each sample in the training set
        for data in dummy_train_dataset_loader:
            pass
        # end for

        # For each sample in the dev set
        for data in dummy_dev_dataset_loader:
            pass
        # end for

        # For each sample in the test set
        for data in dummy_test_dataset_loader:
            pass
        # end for
    # endregion PUBLIC

    # region TEST

    # Test : 10 fold cross validation with no validation set
    def test_10fold_cross_validation_no_dev(self):
        """
        Test : 10 fold cross validation with no validation set.
        """
        # Test parameters
        sample_len = 10
        input_dim = 1
        n_classes = 10
        n_samples = 100

        # Basic case
        self.fold_cross_validation(
            sample_len=sample_len,
            input_dim=input_dim,
            n_classes=n_classes,
            n_samples=n_samples,
            n_folds=10,
            with_dev=False,
            shuffle=False
        )
    # end test_10fold_cross_validation_no_dev

    # endregion TEST

    # endregion BODY
# end Test_Memory_Management
