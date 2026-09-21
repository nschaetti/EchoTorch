# -*- coding: utf-8 -*-
#
# File : test/test_timetensors_constructor.py
# Description : Test the TimeTensor constructor and properties.
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


class Test_TimeTensors_Constructor(EchoTorchTestCase):
    """Test the TimeTensor constructor and property accessors."""

    def test_constructor_time_dim_0(self):
        tt = TimeTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), time_dim=0)

        self.assertEqual(tt.time_dim, 0)
        self.assertEqual(tt.tlen, 2)
        self.assertEqual(tt.cdim, 1)
        self.assertTensorSize(tt.tensor, [2, 2])

    def test_constructor_time_dim_1(self):
        tt = TimeTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), time_dim=1)

        self.assertEqual(tt.time_dim, 1)
        self.assertEqual(tt.tlen, 2)
        self.assertEqual(tt.cdim, 0)
        self.assertTensorSize(tt.tensor, [2, 2])

    def test_constructor_time_dim_middle(self):
        tt = TimeTensor(torch.ones(4, 5, 6), time_dim=1)

        self.assertEqual(tt.time_dim, 1)
        self.assertEqual(tt.tlen, 5)
        self.assertEqual(tt.cdim, 1)
        self.assertTensorSize(tt.tensor, [4, 5, 6])

    def test_constructor_time_dim_too_large_raises(self):
        with self.assertRaises(ValueError):
            TimeTensor(torch.ones(2, 2), time_dim=2)
        # end with

    def test_constructor_from_timetensor_copies_data(self):
        source = TimeTensor(torch.tensor([[1.0], [2.0]]), time_dim=0)
        copy = TimeTensor(source, time_dim=0)

        self.assertEqual(copy.time_dim, source.time_dim)
        self.assertTensorEqual(copy.tensor, source.tensor)

    def test_time_dim_setter_valid(self):
        tt = TimeTensor(torch.ones(2, 3, 4), time_dim=0)
        tt.time_dim = 1

        self.assertEqual(tt.time_dim, 1)

    def test_time_dim_setter_too_large_raises(self):
        tt = TimeTensor(torch.ones(2, 3), time_dim=0)

        with self.assertRaises(ValueError):
            tt.time_dim = 2
        # end with

    def test_time_dim_setter_negative_raises(self):
        tt = TimeTensor(torch.ones(2, 3), time_dim=0)

        with self.assertRaises(ValueError):
            tt.time_dim = -1
        # end with

    def test_cdim_0d(self):
        tt = TimeTensor(torch.tensor([1.0, 2.0, 3.0]), time_dim=0)

        self.assertEqual(tt.cdim, 0)
        self.assertEqual(tt.tlen, 3)

    def test_numelc(self):
        tt = TimeTensor(torch.ones(4, 3, 2), time_dim=0)

        self.assertEqual(tt.numelc(), 3 * 2)

    def test_tlen_equals_time_dim_size(self):
        tt = TimeTensor(torch.ones(7, 5), time_dim=1)

        self.assertEqual(tt.tlen, 5)

    def test_to_same_config_returns_same_object(self):
        tt = TimeTensor(torch.ones(2, 3), time_dim=0)

        self.assertIs(tt.to(dtype=torch.float32), tt)

    def test_equality_reflexive(self):
        tt = TimeTensor(torch.ones(2, 3), time_dim=0)

        self.assertTrue(tt == tt)
        self.assertFalse(tt != tt)


# end Test_TimeTensors_Constructor
