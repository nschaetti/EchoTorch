# -*- coding: utf-8 -*-
#
# File : test/test_timetensors_allowlists.py
# Description : Test the torch ops allowlists used by TimeTensor dispatch.
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
from echotorch.timetensors import TORCH_OPS_IMPLEMENTED, TORCH_OPS_UNSUPPORTED


class Test_TimeTensors_Allowlists(EchoTorchTestCase):
    """Test that the torch ops allowlists recognize the previously-misspelled entries."""

    def test_bitwise_not_is_recognized(self):
        self.assertIn('bitwise_not', TORCH_OPS_IMPLEMENTED)
        self.assertNotIn('bitwose_not', TORCH_OPS_IMPLEMENTED)
    # end test_bitwise_not_is_recognized

    def test_greater_is_recognized(self):
        self.assertIn('greater', TORCH_OPS_IMPLEMENTED)
        self.assertNotIn('greated', TORCH_OPS_IMPLEMENTED)
    # end test_greater_is_recognized

    def test_conv_transpose_is_recognized(self):
        self.assertIn('conv_transpose1d', TORCH_OPS_IMPLEMENTED)
        self.assertIn('conv_transpose2d', TORCH_OPS_IMPLEMENTED)
        self.assertIn('conv_transpose3d', TORCH_OPS_IMPLEMENTED)
        self.assertNotIn('conv_tranpose1d', TORCH_OPS_IMPLEMENTED)
    # end test_conv_transpose_is_recognized

    def test_bitwise_not_dispatch(self):
        tt = TimeTensor(torch.tensor([[0, 1, 0], [1, 0, 1]]), time_dim=0)

        out = torch.bitwise_not(tt)
        self.assertTrue(isinstance(out, TimeTensor))
        self.assertTensorEqual(out.tensor, torch.bitwise_not(torch.tensor([[0, 1, 0], [1, 0, 1]])))
        self.assertEqual(out.time_dim, 0)
    # end test_bitwise_not_dispatch

    def test_greater_dispatch(self):
        tt = TimeTensor(torch.tensor([[1, 2], [3, 4]]), time_dim=0)

        out = torch.greater(tt.tensor, 2)
        self.assertTensorEqual(out, torch.tensor([[False, False], [True, True]]))
    # end test_greater_dispatch

    def test_compare_direct_ops(self):
        tt = TimeTensor(torch.tensor([[1, 2], [3, 4]]), time_dim=0)

        self.assertTrue(tt.allclose(tt))
    # end test_compare_direct_ops

    def test_unsupported_op_is_listed(self):
        self.assertIn('affine_grid', TORCH_OPS_UNSUPPORTED)
    # end test_unsupported_op_is_listed


# end Test_TimeTensors_Allowlists
