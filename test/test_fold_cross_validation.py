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
from . import EchoTorchTestCase
import numpy as np
import torch
import echotorch.evaluation as val
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

        # Tensor for each fold and sets to keep classes
        cross_results_train = list()
        cross_results_dev = list()
        cross_results_test = list()

        # For each fold
        for k in range(n_folds):
            # Set fold
            dummy_cv_train_dataset.set_fold(k)
            dummy_cv_dev_dataset.set_fold(k)
            dummy_cv_test_dataset.set_fold(k)

            # For each sample in the training set
            fold_cross_results_train = torch.zeros(len(dummy_cv_train_dataset))
            for data_i, data in enumerate(dummy_train_dataset_loader):
                # Data
                _, sample_class = data
                fold_cross_results_train[data_i] = sample_class.item()
            # end for

            # For each sample in the dev set
            fold_cross_results_dev = torch.zeros(len(dummy_cv_dev_dataset))
            for data_i, data in enumerate(dummy_dev_dataset_loader):
                # Data
                _, sample_class = data
                fold_cross_results_dev[data_i] = sample_class.item()
            # end for

            # For each sample in the test set
            fold_cross_results_test = torch.zeros(len(dummy_cv_test_dataset))
            for data_i, data in enumerate(dummy_test_dataset_loader):
                # Data
                _, sample_class = data
                fold_cross_results_test[data_i] = sample_class.item()
            # end for

            # Add to results
            cross_results_train.append(fold_cross_results_train)
            cross_results_dev.append(fold_cross_results_dev)
            cross_results_test.append(fold_cross_results_test)
        # end for

        return dummy_cv_train_dataset.folds, dummy_cv_train_dataset.fold_sizes, cross_results_train, \
               cross_results_dev, cross_results_test
    # endregion PUBLIC

    # region TEST

    # Test : 10-fold cross validation, no dev, samples indices specified
    def test_10fold_cross_validation_no_dev_sample_indices_specified(self):
        """
        Test : 10-fold cross validation, no dev, samples indices specified
        """
        # Test parameters
        sample_len = 10
        input_dim = 1
        n_classes = 10
        n_samples = 10

        # Basic case
        folds, fold_sizes, results_train, results_dev, results_test = self.fold_cross_validation(
            sample_len=sample_len,
            input_dim=input_dim,
            n_classes=n_classes,
            n_samples=n_samples,
            n_folds=10,
            with_dev=False,
            shuffle=False,
            sample_indices=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            train_size=1.0,
            shuffle_cv=False
        )

        # Check folds
        for f in range(10):
            self.assertEqual(folds[f][0], 9-f)
        # end for

        # Validate training set
        for i in range(10):
            # All indices
            all_indices = [x for x in range(10)]

            # Selector
            all_indices.remove(folds[i][0])

            # The two must be equal
            self.assertTensorEqual(torch.tensor(all_indices), results_train[i])
        # end for
    # end test_10fold_cross_validation_no_dev_sample_indices_specified

    # Test : 10 fold cross validation with dataset size not multiple of 10
    def test_10fold_cross_validation_not_multiple(self):
        """
        Test : 10-fold cross validation with dataset size not multiple of 10.
        """
        # Test parameters
        sample_len = 10
        input_dim = 1
        n_classes = 97
        n_samples = 97

        # Basic case
        folds, fold_sizes, results_train, results_dev, results_test = self.fold_cross_validation(
            sample_len=sample_len,
            input_dim=input_dim,
            n_classes=n_classes,
            n_samples=n_samples,
            n_folds=10,
            with_dev=False,
            shuffle=False,
            sample_indices=None,
            train_size=1.0,
            shuffle_cv=False
        )

        # Validate fold sizes
        self.assertArrayEqual(np.array(fold_sizes), np.array([10, 10, 10, 10, 10, 10, 10, 9 ,9 ,9]))
    # end test_10fold_cross_validation_not_multiple

    # Test : 10 fold cross validation, no overlapping
    def test_10fold_cross_validation_no_overlapping(self):
        """
        Test : 10 fold cross validation, no overlapping
        """
        # Test parameters
        sample_len = 10
        input_dim = 1
        n_classes = 100
        n_samples = 100

        # Basic case
        folds, fold_sizes, results_train, results_dev, results_test = self.fold_cross_validation(
            sample_len=sample_len,
            input_dim=input_dim,
            n_classes=n_classes,
            n_samples=n_samples,
            n_folds=10,
            with_dev=False,
            shuffle=False,
            sample_indices=None,
            train_size=1.0,
            shuffle_cv=False
        )

        # Test each fold
        for f in range(10):
            # Training and test sets
            training_set = results_train[f]
            test_set = results_test[f]

            # For each element in the training set
            for i in range(training_set.size(0)):
                # Check that this element is not in the test set
                for j in range(test_set.size(0)):
                    self.assertNotEqual(training_set[i].item(), test_set[j].item())
                # end for
            # end for
        # end for
    # end test_10fold_cross_validation_no_overlapping

    # Test : 10 fold cross validation with dev
    def test_10fold_cross_validation_with_dev(self):
        """
        Test : 10-fold cross validation with dev
        """
        # Test parameters
        sample_len = 10
        input_dim = 1
        n_classes = 20
        n_samples = 20

        # Basic case
        folds, fold_sizes, results_train, results_dev, results_test = self.fold_cross_validation(
            sample_len=sample_len,
            input_dim=input_dim,
            n_classes=n_classes,
            n_samples=n_samples,
            n_folds=10,
            with_dev=True,
            shuffle=False,
            sample_indices=None,
            train_size=1.0,
            shuffle_cv=False
        )

        # Validate folds
        for i in range(10):
            self.assertArrayEqual(folds[i], np.array([i*2, i*2+1]))
        # end for

        # Validate fold sizes
        for i in range(10):
            self.assertEqual(fold_sizes[i], 2)
        # end for

        # Validate training set
        for i in range(10):
            # All indices
            all_indices = [x for x in range(20)]

            # Selector
            all_indices.remove(folds[i][0])
            all_indices.remove(folds[i][1])

            # The two must be equal
            self.assertTensorEqual(torch.tensor(all_indices), results_train[i])
        # end for

        # Validate the dev. set
        for i in range(10):
            self.assertEqual(results_dev[i].item(), i*2)
        # end for

        # Validate test set
        for i in range(10):
            self.assertEqual(results_test[i].item(), i*2+1)
        # end for
    # end test_10fold_cross_validation_with_dev

    # Test : 10 fold cross validation with no validation set
    def test_10fold_cross_validation_no_dev(self):
        """
        Test : 10 fold cross validation with no validation set.
        """
        # Test parameters
        sample_len = 10
        input_dim = 1
        n_classes = 10
        n_samples = 10

        # Basic case
        folds, fold_sizes, results_train, results_dev, results_test = self.fold_cross_validation(
            sample_len=sample_len,
            input_dim=input_dim,
            n_classes=n_classes,
            n_samples=n_samples,
            n_folds=10,
            with_dev=False,
            shuffle=False,
            sample_indices=None,
            train_size=1.0,
            shuffle_cv=False
        )

        # Validate folds
        for i in range(10):
            self.assertEqual(folds[i][0], i)
        # end for

        # Validate fold sizes
        for i in range(10):
            self.assertEqual(fold_sizes[i], 1)
        # end for

        # Validate train
        self.assertTensorEqual(results_train[0], torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.]))
        self.assertTensorEqual(results_train[1], torch.tensor([0., 2., 3., 4., 5., 6., 7., 8., 9.]))
        self.assertTensorEqual(results_train[2], torch.tensor([0., 1., 3., 4., 5., 6., 7., 8., 9.]))
        self.assertTensorEqual(results_train[3], torch.tensor([0., 1., 2., 4., 5., 6., 7., 8., 9.]))
        self.assertTensorEqual(results_train[4], torch.tensor([0., 1., 2., 3., 5., 6., 7., 8., 9.]))
        self.assertTensorEqual(results_train[5], torch.tensor([0., 1., 2., 3., 4., 6., 7., 8., 9.]))
        self.assertTensorEqual(results_train[6], torch.tensor([0., 1., 2., 3., 4., 5., 7., 8., 9.]))
        self.assertTensorEqual(results_train[7], torch.tensor([0., 1., 2., 3., 4., 5., 6., 8., 9.]))
        self.assertTensorEqual(results_train[8], torch.tensor([0., 1., 2., 3., 4., 5., 6., 7., 9.]))
        self.assertTensorEqual(results_train[9], torch.tensor([0., 1., 2., 3., 4., 5., 6., 7., 8.]))

        # Validate dev
        for i in range(10):
            self.assertTensorEqual(results_dev[i], torch.tensor([]))
        # end for

        # Validate train
        for i in range(10):
            self.assertTensorEqual(results_test[i], torch.tensor([float(i)]))
        # end for
    # end test_10fold_cross_validation_no_dev

    # endregion TEST

    # endregion BODY
# end Test_Memory_Management
