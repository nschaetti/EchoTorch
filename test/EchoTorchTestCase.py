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


# Test case base class.
class EchoTorchTestCase(TestCase):
    """
    Test case base class
    """

    #########################
    # PUBLIC
    #########################

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

# end EchoTorchTestCase
