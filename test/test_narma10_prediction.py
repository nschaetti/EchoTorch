# -*- coding: utf-8 -*-
#
# File : test/narma10_prediction
# Description : NARMA-10 prediction test case.
# Date : 26th of January, 2018
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
import unittest
from unittest import TestCase


# Test case : NARMA10 timeseries prediction.
class Test_NARMA10_Prediction(TestCase):
    """
    Test NARMA10 timeseries prediction
    """

    ##############################
    # TESTS
    ##############################

    # Simple test
    def test_simple(self):
        """
        Simple test
        :return:
        """
        return True
    # end test_simple

# end Test_NARMA10_Prediction


# Run test
if __name__ == '__main__':
    unittest.main()
# end if
