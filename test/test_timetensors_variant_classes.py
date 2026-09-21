# -*- coding: utf-8 -*-
#
# File : test/test_timetensors_variant_classes.py
# Description : Test the typed variant subclasses of BaseTensor.
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

from echotorch.base_tensors import (
    BaseTensor, DoubleBaseTensor, FloatBaseTensor, HalfBaseTensor, BFloat16Tensor,
    BoolBaseTensor, ByteBaseTensor, CharBaseTensor, ShortBaseTensor,
    IntBaseTensor, LongBaseTensor,
)


class Test_TimeTensors_VariantClasses(EchoTorchTestCase):
    """Test that each typed variant subclass casts its data to the expected dtype."""

    def test_double_base_tensor(self):
        bt = DoubleBaseTensor(torch.tensor([1, 2, 3]))
        self.assertEqual(bt.tensor.dtype, torch.float64)
        self.assertTensorEqual(bt.tensor, torch.tensor([1.0, 2.0, 3.0]))
    # end test_double_base_tensor

    def test_float_base_tensor(self):
        bt = FloatBaseTensor(torch.tensor([1, 2, 3]))
        self.assertEqual(bt.tensor.dtype, torch.float32)
    # end test_float_base_tensor

    def test_half_base_tensor(self):
        bt = HalfBaseTensor(torch.tensor([1, 2, 3]))
        self.assertEqual(bt.tensor.dtype, torch.float16)
    # end test_half_base_tensor

    def test_bfloat16_tensor(self):
        bt = BFloat16Tensor(torch.tensor([1, 2, 3]))
        self.assertEqual(bt.tensor.dtype, torch.bfloat16)
    # end test_bfloat16_tensor

    def test_bool_base_tensor(self):
        bt = BoolBaseTensor(torch.tensor([0, 1, 0]))
        self.assertEqual(bt.tensor.dtype, torch.bool)
    # end test_bool_base_tensor

    def test_byte_base_tensor(self):
        bt = ByteBaseTensor(torch.tensor([1, 2, 3]))
        self.assertEqual(bt.tensor.dtype, torch.uint8)
    # end test_byte_base_tensor

    def test_char_base_tensor(self):
        bt = CharBaseTensor(torch.tensor([1, 2, 3]))
        self.assertEqual(bt.tensor.dtype, torch.int8)
    # end test_char_base_tensor

    def test_short_base_tensor(self):
        bt = ShortBaseTensor(torch.tensor([1, 2, 3]))
        self.assertEqual(bt.tensor.dtype, torch.int16)
    # end test_short_base_tensor

    def test_int_base_tensor(self):
        bt = IntBaseTensor(torch.tensor([1, 2, 3]))
        self.assertEqual(bt.tensor.dtype, torch.int32)
    # end test_int_base_tensor

    def test_long_base_tensor(self):
        bt = LongBaseTensor(torch.tensor([1, 2, 3]))
        self.assertEqual(bt.tensor.dtype, torch.int64)
    # end test_long_base_tensor

    def test_variants_are_base_tensor_instances(self):
        self.assertIsInstance(DoubleBaseTensor(torch.tensor([1.0])), BaseTensor)
    # end test_variants_are_base_tensor_instances


# end Test_TimeTensors_VariantClasses
