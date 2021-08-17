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
import echotorch.utils.matrix_generation as mg
import echotorch.utils
import torch
import numpy as np

# Local imports
from . import EchoTorchTestCase


# Test cases : Test statistical operations on TimeTensors
class Test_TimeTensors_StatOps(EchoTorchTestCase):
    """
    Test cases : Test statistical operations on TimeTensors
    """

    # region TESTS

    # Test covariance

    # endregion TESTS

# end Test_TimeTensors_StatOps

