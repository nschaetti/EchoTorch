# -*- coding: utf-8 -*-
#
# File : echotorch/evaluation/CrossValidationWithDev.py
# Description : Cross validation with dev set.
# Date : 20th of July, 2020
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
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>,
# Université de Genève <nils.schaetti@unige.ch>

# Imports
import math
from torch.utils.data.dataset import Dataset
import numpy as np


# Do a k-fold cross validation with a dev set on a data set
class CrossValidationWithDev(Dataset):
    """
    Do K-fold cross validation with a dev set on a data set
    """

    # Constructor
    def __init__(self, root_dataset, k=10, mode='train', samples_indices=None, fold=0, train_size=1.0, dev_ratio=0.5,
                 shuffle=False):
        """
        Constructor
        :param root_dataset: The target data set
        :param k: Number of fold
        :param train: Return training or test set?
        """
        # Properties
        self.root_dataset = root_dataset
        self.k = k
        self.mode = mode
        self.train_size = train_size
        self.fold = fold
        self.dev_ratio = dev_ratio
        self.shuffle = shuffle

        # Compute fold sizes
        self.folds, self.fold_sizes, self.indexes = self._create_folds(self.k, samples_indices)
    # end __init__

    #region PUBLIC

    # Set in train mode
    def train(self):
        """
        Set in train mode
        """
        self.mode = 'train'
    # end train

    # Set in dev mode
    def dev(self):
        """
        Set in dev mode
        """
        self.mode = 'dev'
    # end dev

    # Set in test mode
    def test(self):
        """
        Set in test mode
        """
        self.mode = 'test'
    # end test

    # Next fold
    def next_fold(self):
        """
        Next fold
        :return:
        """
        self.fold += 1
    # end next_fold

    # Set fold
    def set_fold(self, fold):
        """
        Set fold
        :param fold:
        :return:
        """
        self.fold = fold
    # end set_fold

    # Set size
    def set_size(self, size):
        """
        Set size
        :param size:
        :return:
        """
        self.train_size = size
    # end set_size

    #endregion PUBLIC

    #region PRIVATE

    # Create folds
    def _create_folds(self, k, samples_indices=None):
        """
        Create folds
        :return:
        """
        # Indexes
        if samples_indices is None:
            # Indices
            indexes = np.arange(0, len(self.root_dataset))

            # Shuffle index list
            if self.shuffle:
                np.random.shuffle(indexes)
            # end if
        else:
            indexes = samples_indices
        # end if

        # Dataset length
        length = len(indexes)

        # Division and rest
        division = int(math.floor(length / k))
        reste = length - division * k
        reste_size = k - reste

        # Folds size
        fold_sizes = [division+1] * (reste) + [division] * (reste_size)

        # Folds
        folds = list()
        start = 0
        for i in range(k):
            folds.append(indexes[start:start+fold_sizes[i]])
            start += fold_sizes[i]
        # end for

        return folds, fold_sizes, indexes
    # end _create_folds

    #endregion PRIVATE

    #region OVERRIDE

    # Dataset size
    def __len__(self):
        """
        Dataset size
        :return:
        """
        # Test length
        dev_test_length = self.fold_sizes[self.fold]
        train_length = len(self.root_dataset) - dev_test_length
        dev_length = int(dev_test_length * self.dev_ratio)
        test_length = dev_test_length - dev_length

        if self.mode == 'train':
            return int(train_length * self.train_size)
        elif self.mode == 'dev':
            return dev_length
        else:
            return test_length
        # end if
    # end __len__

    # Get item
    def __getitem__(self, item):
        """
        Get item
        :param item:
        :return:
        """
        # Get target set
        dev_test_set = self.folds[self.fold]
        indexes_copy = self.indexes.copy()
        train_set = np.setdiff1d(indexes_copy, dev_test_set)
        train_length = len(self.root_dataset) - len(dev_test_set)
        train_length = int(train_length * self.train_size)
        train_set = train_set[:train_length]

        # Dev/test length
        dev_length = int(len(dev_test_set) * self.dev_ratio)

        # Dev/test sets
        dev_set = dev_test_set[:dev_length]
        test_set = dev_test_set[dev_length:]

        # Train/test
        if self.mode == 'train':
            return self.root_dataset[train_set[item]]
        elif self.mode == 'dev':
            return self.root_dataset[dev_set[item]]
        else:
            return self.root_dataset[test_set[item]]
        # end if
    # end __getitem__

    #endregion OVERRIDE

# end CrossValidationWithDev
