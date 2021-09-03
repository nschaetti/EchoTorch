# -*- coding: utf-8 -*-
#
# File : echotorch/timetensor.py
# Description : A special tensor with a time dimension
# Date : 25th of January, 2021
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


# Test PyTorch Indexing Ops
class Test_Torch_Ops_Indexing(EchoTorchTestCase):
    r"""Test PyTorch Indexing Ops
    """

    # region TESTS

    # Test cat
    def test_cat(self):
        r"""Test :func:`torch.cat`.
        """
        # Parameters
        time_length = 10
        n_features = 2
        time_dim = 1

        # Tensors and timetensor
        x = torch.randn(time_length, n_features)
        y = echotorch.randn(n_features, length=time_length)
        z = echotorch.as_timetensor(torch.randn(time_length, n_features), time_dim=time_dim)

        # Cat y and z
        out = torch.cat((y, z), 0)
        assert out.size()[0] == time_length * 2 and out.size()[1] == n_features
        assert out.time_dim == 0
        assert isinstance(out, echotorch.TimeTensor)

        # Cat x and z
        out = torch.cat((x, z), 0)
        assert out.size()[0] == time_length * 2 and out.size()[1] == n_features
        assert out.time_dim == 1
        assert isinstance(out, echotorch.TimeTensor)
    # end test_cat

    # Test chunk
    def test_chunk(self):
        r"""Test :func:`torch.chunk`.
        """
        # Parameters
        time_length = 10
        n_features = 2

        # Tensors and TimeTensors
        y = echotorch.randn(n_features, length=time_length)

        out = torch.chunk(y, 2, 0)
        assert len(out) == 2
        assert out[0].size()[0] == 5 and out[0].size()[1] == n_features
        assert isinstance(out[0], echotorch.TimeTensor)
    # end test_chunk

    # dsplit
    def test_dsplit(self):
        r"""Test :func:`torch.dsplit`.
        """
        # Parameters
        time_length = 10
        n_features = 2

        # TimeTensors
        x = torch.arange(16.0).reshape(2, 2, 4)
        y = echotorch.timetensor(x, time_dim=2)
        z = echotorch.timetensor(x, time_dim=1)

        # dsplit y
        out = torch.dsplit(y, [1, 3])
        assert len(out) == 3
        assert out[0].size()[2] == 1
        assert out[0].time_dim == 2
        assert out[1].size()[2] == 2
        assert out[1].time_dim == 2
        assert out[2].size()[2] == 1
        assert out[2].time_dim == 2
        assert isinstance(out[0], echotorch.TimeTensor)

        # dsplit z
        out = torch.dsplit(z, [1, 3])
        assert len(out) == 3
        assert out[0].size()[2] == 1
        assert out[0].time_dim == 1
        assert out[1].size()[2] == 2
        assert out[1].time_dim == 1
        assert out[2].size()[2] == 1
        assert out[2].time_dim == 1
        assert isinstance(out[0], echotorch.TimeTensor)
    # end test_dsplit

    # Test column_stack
    def test_column_stack(self):
        r"""Test :func:`torch.column_stack`.
        """
        # Time length
        time_length = 20
        n_features = 2
        time_dim = 0

        # Tensors/TimeTensors
        x = torch.arange(time_length)
        y = torch.arange(time_length * 2).reshape(time_length, n_features)
        z = echotorch.arange(time_length)

        # Test 1
        out = torch.column_stack((z, y))
        assert out.size()[0] == time_length
        assert out.size()[1] == n_features + 1
        assert out.time_dim == time_dim
        assert isinstance(out, echotorch.TimeTensor)

        # Test 2
        out = torch.column_stack((z, x))
        assert out.size()[0] == time_length
        assert out.size()[1] == 2
        assert out.time_dim == time_dim
        assert isinstance(out, echotorch.TimeTensor)
    # end test_column_stack

    # endregion TESTS

# end Test_Torch_Ops_Indexing
