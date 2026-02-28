# -*- coding: utf-8 -*-
#
# File : test/test_timetensors_stats.py
# Description : Test statistical operations on TimeTensors.
# Date : 17th of August, 2021
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
import echotorch

# Local imports
from . import EchoTorchTestCase


class Test_TimeTensors_StatOps(EchoTorchTestCase):
    """Test statistical and core torch ops on TimeTensors."""

    def test_stat_ops_match_torch(self):
        x = torch.arange(40, dtype=torch.float32).reshape(5, 8)
        tt = echotorch.as_timetensor(x, time_dim=0)

        self.assertTensorAlmostEqual(echotorch.tmean(tt), torch.mean(x, dim=0), 1e-6)
        self.assertTensorAlmostEqual(echotorch.tstd(tt), torch.std(x, dim=0, unbiased=True), 1e-6)
        self.assertTensorAlmostEqual(echotorch.tvar(tt), torch.var(x, dim=0, unbiased=True), 1e-6)

    def test_cat_kwargs_returns_timetensor(self):
        x = echotorch.randn(3, length=7)
        y = echotorch.randn(3, length=7)
        out = torch.cat(tensors=[x, y], dim=x.time_dim)

        self.assertTrue(isinstance(out, echotorch.TimeTensor))
        self.assertEqual(out.time_dim, x.time_dim)
        self.assertTensorSize(out.tensor, [14, 3])

    def test_permute_updates_time_dim(self):
        x = echotorch.rand(4, 3, length=6)
        out = torch.permute(x, (1, 0, 2))

        self.assertTrue(isinstance(out, echotorch.TimeTensor))
        self.assertEqual(out.time_dim, 1)
        self.assertTensorSize(out.tensor, [4, 6, 3])

    def test_flatten_non_time_dims_keeps_timetensor(self):
        x = echotorch.rand(4, 3, length=6)
        out = torch.flatten(x, start_dim=1)

        self.assertTrue(isinstance(out, echotorch.TimeTensor))
        self.assertEqual(out.time_dim, 0)
        self.assertTensorSize(out.tensor, [6, 12])

    def test_reduction_on_time_dim_without_keepdim_returns_tensor(self):
        x = echotorch.rand(4, 3, length=6)

        m = torch.mean(x, dim=x.time_dim)
        s = torch.sum(x, dim=x.time_dim)
        v = torch.var(x, dim=x.time_dim)

        self.assertFalse(isinstance(m, echotorch.TimeTensor))
        self.assertFalse(isinstance(s, echotorch.TimeTensor))
        self.assertFalse(isinstance(v, echotorch.TimeTensor))
        self.assertTensorSize(m, [4, 3])
        self.assertTensorSize(s, [4, 3])
        self.assertTensorSize(v, [4, 3])

    def test_reduction_on_time_dim_with_keepdim_returns_timetensor(self):
        x = echotorch.rand(4, 3, length=6)

        m = torch.mean(x, dim=x.time_dim, keepdim=True)
        s = torch.sum(x, dim=x.time_dim, keepdim=True)

        self.assertTrue(isinstance(m, echotorch.TimeTensor))
        self.assertTrue(isinstance(s, echotorch.TimeTensor))
        self.assertEqual(m.time_dim, x.time_dim)
        self.assertEqual(s.time_dim, x.time_dim)
        self.assertTensorSize(m.tensor, [1, 4, 3])
        self.assertTensorSize(s.tensor, [1, 4, 3])

    def test_reduction_non_time_dim_updates_time_dim(self):
        x = echotorch.rand(4, 3, length=6)

        out_no_keep = torch.mean(x, dim=1, keepdim=False)
        out_keep = torch.mean(x, dim=1, keepdim=True)

        self.assertTrue(isinstance(out_no_keep, echotorch.TimeTensor))
        self.assertTrue(isinstance(out_keep, echotorch.TimeTensor))
        self.assertEqual(out_no_keep.time_dim, 0)
        self.assertEqual(out_keep.time_dim, 0)
        self.assertTensorSize(out_no_keep.tensor, [6, 3])
        self.assertTensorSize(out_keep.tensor, [6, 1, 3])

    def test_reduction_multiple_dims_with_time_dim_returns_tensor(self):
        x = echotorch.rand(4, 3, length=6)
        out = torch.sum(x, dim=(0, 1))

        self.assertFalse(isinstance(out, echotorch.TimeTensor))
        self.assertTensorSize(out, [3])

    def test_max_with_dim_returns_timetensor_pair_when_time_not_reduced(self):
        x = echotorch.rand(4, 3, length=6)
        values, indices = torch.max(x, dim=1)

        self.assertTrue(isinstance(values, echotorch.TimeTensor))
        self.assertTrue(isinstance(indices, echotorch.TimeTensor))
        self.assertEqual(values.time_dim, 0)
        self.assertEqual(indices.time_dim, 0)
        self.assertTensorSize(values.tensor, [6, 3])
        self.assertTensorSize(indices.tensor, [6, 3])

    def test_reduction_negative_dim_keeps_timetensor(self):
        x = echotorch.rand(4, 3, length=6)
        out = torch.mean(x, dim=-1)

        self.assertTrue(isinstance(out, echotorch.TimeTensor))
        self.assertEqual(out.time_dim, 0)
        self.assertTensorSize(out.tensor, [6, 4])


# end Test_TimeTensors_StatOps
