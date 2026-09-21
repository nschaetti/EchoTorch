# -*- coding: utf-8 -*-
#
# File : test/test_timetensors_arithmetic.py
# Description : Test the arithmetic operators of BaseTensor/TimeTensor.
# Date : 3rd of September, 2026
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
import torch

# Local imports
from . import EchoTorchTestCase

from echotorch.timetensors import TimeTensor
from echotorch.base_tensors import BaseTensor


class Test_TimeTensors_Arithmetic(EchoTorchTestCase):
    """Test that arithmetic operators return new objects and leave the original unmodified."""

    def setUp(self):
        self.a = TimeTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), time_dim=0)
        self.orig = self.a.tensor.clone()
    # end setUp

    def assert_unmodified(self):
        self.assertTensorEqual(self.a.tensor, self.orig)
    # end assert_unmodified

    def test_add_scalar_returns_new(self):
        b = self.a + 10.0
        self.assertIsNot(b, self.a)
        self.assertTrue(isinstance(b, TimeTensor))
        self.assertTensorAlmostEqual(b.tensor, self.orig + 10.0, 1e-6)
        self.assert_unmodified()
    # end test_add_scalar_returns_new

    def test_radd_scalar_returns_new(self):
        b = 10.0 + self.a
        self.assertIsNot(b, self.a)
        self.assertTensorAlmostEqual(b.tensor, self.orig + 10.0, 1e-6)
        self.assert_unmodified()
    # end test_radd_scalar_returns_new

    def test_sub_scalar_returns_new(self):
        b = self.a - 10.0
        self.assertIsNot(b, self.a)
        self.assertTensorAlmostEqual(b.tensor, self.orig - 10.0, 1e-6)
        self.assert_unmodified()
    # end test_sub_scalar_returns_new

    def test_rsub_scalar_returns_new(self):
        b = 10.0 - self.a
        self.assertIsNot(b, self.a)
        self.assertTensorAlmostEqual(b.tensor, 10.0 - self.orig, 1e-6)
        self.assert_unmodified()
    # end test_rsub_scalar_returns_new

    def test_mul_scalar_returns_new(self):
        b = self.a * 3.0
        self.assertIsNot(b, self.a)
        self.assertTensorAlmostEqual(b.tensor, self.orig * 3.0, 1e-6)
        self.assert_unmodified()
    # end test_mul_scalar_returns_new

    def test_rmul_scalar_returns_new(self):
        b = 3.0 * self.a
        self.assertIsNot(b, self.a)
        self.assertTensorAlmostEqual(b.tensor, self.orig * 3.0, 1e-6)
        self.assert_unmodified()
    # end test_rmul_scalar_returns_new

    def test_truediv_scalar_returns_new(self):
        b = self.a / 2.0
        self.assertIsNot(b, self.a)
        self.assertTensorAlmostEqual(b.tensor, self.orig / 2.0, 1e-6)
        self.assert_unmodified()
    # end test_truediv_scalar_returns_new

    def test_add_timetensor_returns_new(self):
        b = self.a + self.a
        self.assertIsNot(b, self.a)
        self.assertTensorAlmostEqual(b.tensor, self.orig + self.orig, 1e-6)
        self.assert_unmodified()
    # end test_add_timetensor_returns_new

    def test_add_base_tensor_returns_new(self):
        base = BaseTensor(self.orig)
        b = self.a + base
        self.assertTensorAlmostEqual(b.tensor, self.orig + self.orig, 1e-6)
        self.assert_unmodified()
    # end test_add_base_tensor_returns_new

    def test_operators_preserve_time_dim(self):
        b = (self.a + 1.0) * 2.0 - 1.0
        self.assertEqual(b.time_dim, self.a.time_dim)
        self.assertEqual(b.tlen, self.a.tlen)
    # end test_operators_preserve_time_dim

    def test_list_sum_uses_radd(self):
        total = sum([TimeTensor(torch.ones(2, 1) * v, time_dim=0) for v in (1.0, 2.0, 3.0)])
        self.assertTensorAlmostEqual(total.tensor, torch.ones(2, 1) * 6.0, 1e-6)
    # end test_list_sum_uses_radd


# end Test_TimeTensors_Arithmetic
