# -*- coding: utf-8 -*-
#
# File : examples/MNIST/convert_images.py
# Description : Convert images to time series.
# Date : 6th of April, 2017
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

"""
Created on 6 April 2017
@author: Nils Schaetti
"""

import sys
import os
sys.path.insert(0, os.path.abspath('./../..'))
import echotorch


if __name__ == "__main__":

    converter = echotorch.dataset.ImageConverter()

# end if
