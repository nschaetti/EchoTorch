# -*- coding: utf-8 -*-
#
# File : test/test_memory_management.py
# Description : Test incremental reservoir loading and output learning, quota and generation.
# Date : 3th of November, 2019
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
import os
import echotorch.utils
from .EchoTorchTestCase import EchoTorchTestCase
import numpy as np
import torch
import echotorch.nn.conceptors as ecnc
import echotorch.utils.matrix_generation as mg
import echotorch.utils
import echotorch.datasets as etds
from echotorch.datasets import DatasetComposer
from echotorch.nn.Node import Node
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable


# Test case : incremental loading and memory management
class Test_Memory_Management(EchoTorchTestCase):
    """
    Incremental loading and memory management
    """
    # region BODY

    # region PUBLIC

    # Memory management
    def memory_management(self, data_dir, reservoir_size=100, spectral_radius=1.5, input_scaling=1.5, bias_scaling=0.2,
                            connectivity=10.0, washout_length=500, learn_length=1000, ridge_param_wstar=0.0001,
                            ridge_param_wout=0.01, aperture=10, precision=0.001, torch_seed=1, np_seed=1):
        """
        Subspace first demo
        :param data_dir: Directory in the test directory
        :param reservoir_size:
        :param spectral_radius:
        :param input_scaling:
        :param bias_scaling:
        :param connectivity:
        :param washout_length:
        :param learn_length:
        :param ridge_param_wstar:
        :param ridge_param_wout:
        :param aperture:
        :param precision:
        :param torch_seed:
        :param np_seed:
        :return:
        """
        # Package
        subpackage_dir, this_filename = os.path.split(__file__)
        package_dir = os.path.join(subpackage_dir, "..")
        TEST_PATH = os.path.join(package_dir, "data", "tests", data_dir)
    # end memory_management

    # endregion PUBLIC

    # region TEST

    # Memory management
    def test_memory_management(self):
        """
        Memory management
        """
        self.memory_management(data_dir="memory_management")
    # end test_memory_management

    # endregion TEST

    # endregion BODY
# end Test_Memory_Management
