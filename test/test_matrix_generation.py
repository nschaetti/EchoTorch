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


# Test case : Matrix generation
class Test_Matrix_Generation(EchoTorchTestCase):
    """
    Test matrix generation
    """

    #region PUBLIC

    #endregion PUBLIC

    #region PRIVATE
    #endregion PRIVATE

    #region TESTS

    # Test generation of normal matrix
    def test_normal_matrix_generation(self):
        """
        Test generation of normal matrix
        """
        # Set seeds
        torch.manual_seed(1.0)

        # Normal matrix generator
        matrix_generator = mg.NormalMatrixGenerator(
            connectivity=0.1,
            scale=1.0,
            spectral_radius=0.99,
            minimum_edges=4
        )

        # Generate three matrices
        matrix1 = matrix_generator.generate(size=(50, 50))
        matrix2 = matrix_generator.generate(size=(10, 10))
        matrix3 = matrix_generator.generate(size=(2, 2))

        # Test values
        self.assertAlmostEqual(matrix1[1, 3].item(), 0.7379, places=1)
        self.assertAlmostEqual(matrix1[1, 17].item(), -0.2756, places=1)
        self.assertAlmostEqual(matrix1[49, 6].item(), -0.1838, places=1)
        self.assertAlmostEqual(matrix1[49, 17].item(), -0.6171, places=1)
        self.assertAlmostEqual(matrix2[2, 2].item(), -0.4022, places=1)
        self.assertAlmostEqual(matrix2[9, 3].item(), 0.0580, places=1)
        self.assertAlmostEqual(matrix3[0, 0].item(), -0.3698, places=1)
        self.assertAlmostEqual(matrix3[0, 1].item(), -1.1373, places=1)
        self.assertAlmostEqual(matrix3[1, 0].item(), 1.0411242763194695, places=1)
        self.assertAlmostEqual(matrix3[1, 1].item(), -1.6102, places=1)
    # end test_normal_matrix_generation

    #endregion TESTS

# end Test_Matrix_Generation
