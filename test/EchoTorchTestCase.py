# -*- coding: utf-8 -*-
#
# File : test/EchoTorchTestCase
# Description : EchoTorch's test case, with additional utility functions.
# Date : Third of November, 2019
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
from unittest import TestCase
import torch
import numpy as np
import numpy.linalg as lin


# Test case base class.
class EchoTorchTestCase(TestCase):
    """
    Test case base class
    """

    # region PUBLIC

    # Numpy array almost equal with Frobenius norm
    def assertArrayAlmostEqual(self, array1, array2, precision):
        """
        Numpy array almost equal
        :param array1: First array to test
        :param array2: Second array to test
        :param precision: Minimum precision
        """
        # Frob. norm difference
        norm_diff = float(lin.norm(array1 - array2))

        # Places
        places = int(-np.log10(precision))

        # Test precision
        self.assertAlmostEqual(
            norm_diff,
            0.0,
            places
        )
    # end assertArrayAlmostEqual

    # Numpy array equal
    def assertArrayEqual(self, array1, array2):
        """
        Numpy array equal
        :param array1: First array to test
        :param array2: Second array to test
        """
        # Check sizes
        self.assertEqual(array1.shape, array2.shape)

        # Check values
        assert np.all(np.equal(array1, array2))
    # end assertArrayAlmostEqual

    # Assert that two tensors are equal
    def assertTensorEqual(self, tensor1, tensor2):
        """
        Tensor equal (by values)
        :param tensor1: First tensor to compare
        :param tensor2: Second tensor to compare
        """
        # Tensor have equal sizes
        self.assertTensorSize(tensor1, list(tensor2.size()))

        # Check that two tensor are equal
        assert torch.all(torch.eq(tensor1, tensor2))
    # end assertTensorEqual

    # Tensor almost equal with Frobenius norm
    def assertTensorAlmostEqual(self, tensor1, tensor2, precision):
        """
        Tensor almost equation with Frobenius norm
        :param tensor1: Tensor to test
        :param tensor2: Tensor to compare
        :param precision: Minimum precision
        """
        # Compute Frobenius norm difference
        norm_diff = torch.norm(tensor1 - tensor2).item()

        # Places
        places = int(-np.log10(precision))

        # Test precision
        self.assertAlmostEqual(
            norm_diff,
            0.0,
            places
        )
    # end assertTensorAlmostEqual

    # Check the size of a tensor
    def assertTensorSize(self, tensor1, tensor_size):
        """
        Check the size of a tensor
        :param tensor1: Tensor to check
        :param tensor_size: Target tensor size
        """
        self.assertEqual(list(tensor1.size()), tensor_size)
    # endregion PUBLIC

# end EchoTorchTestCase
