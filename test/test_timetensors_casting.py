# -*- coding: utf-8 -*-
#
# File : test/test_timetensors_casting.py
# Description : Test the in-place casting methods of BaseTensor/TimeTensor.
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


class Test_TimeTensors_Casting(EchoTorchTestCase):
    """Test in-place dtype casting methods of a TimeTensor."""

    def setUp(self):
        self.tt = TimeTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), time_dim=0)
    # end setUp

    def test_double(self):
        ret = self.tt.double()
        self.assertIs(ret, self.tt)
        self.assertEqual(self.tt.tensor.dtype, torch.float64)
    # end test_double

    def test_float(self):
        ret = self.tt.float()
        self.assertIs(ret, self.tt)
        self.assertEqual(self.tt.tensor.dtype, torch.float32)
    # end test_float

    def test_half(self):
        ret = self.tt.half()
        self.assertIs(ret, self.tt)
        self.assertEqual(self.tt.tensor.dtype, torch.float16)
    # end test_half

    def test_bfloat16_returns_self_and_changes_dtype(self):
        ret = self.tt.bfloat16()
        self.assertIs(ret, self.tt)
        self.assertEqual(self.tt.tensor.dtype, torch.bfloat16)
    # end test_bfloat16_returns_self_and_changes_dtype

    def test_bool(self):
        ret = self.tt.bool()
        self.assertIs(ret, self.tt)
        self.assertEqual(self.tt.tensor.dtype, torch.bool)
    # end test_bool

    def test_byte(self):
        ret = self.tt.byte()
        self.assertIs(ret, self.tt)
        self.assertEqual(self.tt.tensor.dtype, torch.uint8)
    # end test_byte

    def test_char(self):
        ret = self.tt.char()
        self.assertIs(ret, self.tt)
        self.assertEqual(self.tt.tensor.dtype, torch.int8)
    # end test_char

    def test_short_uses_int16(self):
        ret = self.tt.short()
        self.assertIs(ret, self.tt)
        self.assertEqual(self.tt.tensor.dtype, torch.int16)
    # end test_short_uses_int16

    def test_int(self):
        ret = self.tt.int()
        self.assertIs(ret, self.tt)
        self.assertEqual(self.tt.tensor.dtype, torch.int32)
    # end test_int

    def test_long(self):
        ret = self.tt.long()
        self.assertIs(ret, self.tt)
        self.assertEqual(self.tt.tensor.dtype, torch.int64)
    # end test_long

    def test_casting_preserves_time_dim(self):
        self.tt.double()
        self.assertEqual(self.tt.time_dim, 0)
        self.assertEqual(self.tt.tlen, 2)
    # end test_casting_preserves_time_dim

    def test_to_different_dtype_returns_new_timetensor(self):
        out = self.tt.to(dtype=torch.float64)
        self.assertIsNot(out, self.tt)
        self.assertIsInstance(out, TimeTensor)
        self.assertEqual(out.tensor.dtype, torch.float64)
        self.assertEqual(out.time_dim, 0)
    # end test_to_different_dtype_returns_new_timetensor

    def test_to_same_config_returns_self(self):
        out = self.tt.to(dtype=self.tt.tensor.dtype)
        self.assertIs(out, self.tt)
    # end test_to_same_config_returns_self


# end Test_TimeTensors_Casting
