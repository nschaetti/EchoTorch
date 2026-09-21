# -*- coding: utf-8 -*-
#
# File : test/test_stat_ops.py
# Description : Test the statistical operations on TimeTensors.
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
from echotorch.stat_ops import tquantile, tstd, tvar, cor, cov


class Test_StatOps(EchoTorchTestCase):
    """Test the statistical operations defined for TimeTensors."""

    def test_tquantile_matches_torch(self):
        x = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        tt = TimeTensor(x, time_dim=0)

        self.assertTensorAlmostEqual(tquantile(tt, 0.5), torch.quantile(x, 0.5, dim=0), 1e-6)
        self.assertTensorAlmostEqual(tquantile(tt, 0.25), torch.quantile(x, 0.25, dim=0), 1e-6)
    # end test_tquantile_matches_torch

    def test_tstd_default_correction_1(self):
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        tt = TimeTensor(x, time_dim=0)

        self.assertTensorAlmostEqual(tstd(tt), torch.std(x, dim=0, unbiased=True), 1e-6)
    # end test_tstd_default_correction_1

    def test_tstd_correction_0(self):
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        tt = TimeTensor(x, time_dim=0)

        self.assertTensorAlmostEqual(tstd(tt, correction=0), torch.std(x, dim=0, unbiased=False), 1e-6)
    # end test_tstd_correction_0

    def test_tvar_default_matches_torch(self):
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        tt = TimeTensor(x, time_dim=0)

        self.assertTensorAlmostEqual(tvar(tt), torch.var(x, dim=0, unbiased=True), 1e-6)
        self.assertTensorAlmostEqual(tvar(tt, correction=0), torch.var(x, dim=0, unbiased=False), 1e-6)
    # end test_tvar_default_matches_torch

    def test_cov_autocovariance_without_t2(self):
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        tt = TimeTensor(x, time_dim=0)

        expected = torch.var(x, dim=0, unbiased=True)
        self.assertTensorAlmostEqual(cov(tt), expected, 1e-6)
    # end test_cov_autocovariance_without_t2

    def test_cov_two_series(self):
        a = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        b = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
        tta = TimeTensor(a, time_dim=0)
        ttb = TimeTensor(b, time_dim=0)

        # cov of a series with a scaled copy = 2 * var(a)
        self.assertTensorAlmostEqual(cov(tta, ttb), 2.0 * torch.var(a, dim=0, unbiased=True), 1e-6)
    # end test_cov_two_series

    def test_cov_bias_uses_ddof_0(self):
        a = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        tta = TimeTensor(a, time_dim=0)

        self.assertTensorAlmostEqual(cov(tta, tta, bias=True), torch.var(a, dim=0, unbiased=False), 1e-6)
    # end test_cov_bias_uses_ddof_0

    def test_cov_multivariate(self):
        x = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        tt = TimeTensor(x, time_dim=0)

        c = cov(tt, tt)
        self.assertTensorSize(c, [2, 2])
        self.assertTensorAlmostEqual(c, torch.cov(x.t(), correction=1), 1e-6)
    # end test_cov_multivariate

    def test_cor_autocorrelation_without_t2(self):
        a = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        tta = TimeTensor(a, time_dim=0)

        # Correlation of a series with itself is 1
        self.assertTensorAlmostEqual(cor(tta), torch.ones(1, 1), 1e-6)
    # end test_cor_autocorrelation_without_t2

    def test_cor_perfect_positive(self):
        a = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        b = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
        tta = TimeTensor(a, time_dim=0)
        ttb = TimeTensor(b, time_dim=0)

        self.assertTensorAlmostEqual(cor(tta, ttb), torch.ones(1, 1), 1e-6)
    # end test_cor_perfect_positive

    def test_cor_perfect_negative(self):
        a = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        b = torch.tensor([[-1.0], [-2.0], [-3.0], [-4.0]])
        tta = TimeTensor(a, time_dim=0)
        ttb = TimeTensor(b, time_dim=0)

        self.assertTensorAlmostEqual(cor(tta, ttb), -torch.ones(1, 1), 1e-6)
    # end test_cor_perfect_negative

    def test_cov_and_cor_require_time_first(self):
        tt = TimeTensor(torch.ones(3, 4, 5), time_dim=1)

        with self.assertRaises(ValueError):
            cov(tt)
        # end with

    def test_cov_requires_same_time_length(self):
        tta = TimeTensor(torch.ones(4, 1), time_dim=0)
        ttb = TimeTensor(torch.ones(5, 1), time_dim=0)

        with self.assertRaises(ValueError):
            cov(tta, ttb)
        # end with


# end Test_StatOps
