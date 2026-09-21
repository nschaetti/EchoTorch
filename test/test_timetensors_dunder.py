# -*- coding: utf-8 -*-
#
# File : test/test_timetensors_dunder.py
# Description : Test the dunder (magic) methods of BaseTensor/TimeTensor.
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


class Test_TimeTensors_Dunder(EchoTorchTestCase):
    """Test the dunder methods (comparison and identity semantics)."""

    def test_eq_true_for_same_values(self):
        a = TimeTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), time_dim=0)
        b = TimeTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), time_dim=0)

        self.assertTrue(a == b)

    def test_eq_false_for_different_values(self):
        a = TimeTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), time_dim=0)
        b = TimeTensor(torch.tensor([[5.0, 6.0], [7.0, 8.0]]), time_dim=0)

        self.assertFalse(a == b)

    def test_eq_false_for_different_time_dim(self):
        a = TimeTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), time_dim=0)
        b = TimeTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), time_dim=1)

        self.assertFalse(a == b)

    def test_ne_is_negation_of_eq(self):
        a = TimeTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), time_dim=0)
        same = TimeTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), time_dim=0)
        diff = TimeTensor(torch.tensor([[9.0, 2.0], [3.0, 4.0]]), time_dim=0)

        self.assertFalse(a != same)
        self.assertTrue(a != diff)

    def test_base_tensor_eq_and_ne(self):
        a = BaseTensor(torch.tensor([1.0, 2.0, 3.0]))
        b = BaseTensor(torch.tensor([1.0, 2.0, 3.0]))
        c = BaseTensor(torch.tensor([1.0, 2.0, 4.0]))

        self.assertTrue(a == b)
        self.assertFalse(a == c)
        self.assertFalse(a != b)
        self.assertTrue(a != c)

    def test_gt_ge_lt_le(self):
        a = TimeTensor(torch.tensor([[3.0]]), time_dim=0)

        self.assertTrue((a > 2.0).tensor[0, 0].item())
        self.assertTrue((a >= 3.0).tensor[0, 0].item())
        self.assertTrue((a < 4.0).tensor[0, 0].item())
        self.assertTrue((a <= 3.0).tensor[0, 0].item())


# end Test_TimeTensors_Dunder
