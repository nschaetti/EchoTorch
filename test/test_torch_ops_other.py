# -*- coding: utf-8 -*-
#
# File : test/test_torch_ops_other.py
# Description : Test compatibility with PyTorch operations.
# Date : 3rd of September, 2021
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
from typing import Optional, Tuple, Union, List, Callable, Any
import torch
import echotorch
import numpy as np
import warnings

# Test case
from . import EchoTorchTestCase


# Test PyTorch Other Ops
class Test_Torch_Ops_Other(EchoTorchTestCase):
    r"""Test PyTorch Other Ops.
    """

    # region TESTS

    # Test cummax
    def test_cummax(self):
        r"""Test :func:`torch.cummax`.
        """
        # Parameters
        time_length = 10
        n_features = 4
        time_dim = 0

        # TimeTensor(s)
        x = echotorch.randn(n_features, length=time_length)

        # Test 1
        out = torch.cummax(x, dim=0)
        assert out[0].size()[0] == time_length
        assert out[0].size()[1] == n_features
        assert out[0].time_dim == time_dim
        assert isinstance(out[0], echotorch.TimeTensor)
        assert out[1].size()[0] == time_length
        assert out[1].size()[1] == n_features
        assert out[1].time_dim == time_dim
        assert isinstance(out[1], echotorch.TimeTensor)

        # Test 2
        out = torch.cummax(x, dim=1)
        assert out[0].size()[0] == time_length
        assert out[0].size()[1] == n_features
        assert out[0].time_dim == time_dim
        assert isinstance(out[0], echotorch.TimeTensor)
        assert out[1].size()[0] == time_length
        assert out[1].size()[1] == n_features
        assert out[1].time_dim == time_dim
        assert isinstance(out[1], echotorch.TimeTensor)
    # end test_cummax

    # Test cummin
    def test_cummin(self):
        r"""Test :func:`torch.cummin`.
        """
        # Parameters
        time_length = 10
        n_features = 4
        time_dim = 0

        # TimeTensor(s)
        x = echotorch.randn(n_features, length=time_length)

        # Test 1
        out = torch.cummin(x, dim=0)
        assert out[0].size()[0] == time_length
        assert out[0].size()[1] == n_features
        assert out[0].time_dim == time_dim
        assert isinstance(out[0], echotorch.TimeTensor)
        assert out[1].size()[0] == time_length
        assert out[1].size()[1] == n_features
        assert out[1].time_dim == time_dim
        assert isinstance(out[1], echotorch.TimeTensor)

        # Test 2
        out = torch.cummin(x, dim=1)
        assert out[0].size()[0] == time_length
        assert out[0].size()[1] == n_features
        assert out[0].time_dim == time_dim
        assert isinstance(out[0], echotorch.TimeTensor)
        assert out[1].size()[0] == time_length
        assert out[1].size()[1] == n_features
        assert out[1].time_dim == time_dim
        assert isinstance(out[1], echotorch.TimeTensor)
    # end test_cummin

    # Test cumprod
    def test_cumprod(self):
        r"""Test :func:`torch.cumprod`.
        """
        # Parameters
        time_length = 10
        n_features = 4
        time_dim = 0

        # TimeTensor(s)
        x = echotorch.randn(n_features, length=time_length)

        # Test 1
        out = torch.cumprod(x, dim=0)
        assert out.size()[0] == time_length
        assert out.size()[1] == n_features
        assert out.time_dim == time_dim
        assert isinstance(out, echotorch.TimeTensor)

        # Test 2
        out = torch.cumprod(x, dim=1)
        assert out.size()[0] == time_length
        assert out.size()[1] == n_features
        assert out.time_dim == time_dim
        assert isinstance(out, echotorch.TimeTensor)
    # end test_cumprod

    # Test cumsum
    def test_cumsum(self):
        r"""Test :func:`torch.cumsum`.
        """
        # Parameters
        time_length = 10
        n_features = 4
        time_dim = 0

        # TimeTensor(s)
        x = echotorch.randn(n_features, length=time_length)

        # Test 1
        out = torch.cumsum(x, dim=0)
        assert out.size()[0] == time_length
        assert out.size()[1] == n_features
        assert out.time_dim == time_dim
        assert isinstance(out, echotorch.TimeTensor)

        # Test 2
        out = torch.cumsum(x, dim=1)
        assert out.size()[0] == time_length
        assert out.size()[1] == n_features
        assert out.time_dim == time_dim
        assert isinstance(out, echotorch.TimeTensor)
    # end test_cumsum

    # endregion TESTS

# end Test_Torch_Ops_Other
