# -*- coding: utf-8 -*-
#
# File : examples/timetensor/creation_ops.py
# Description : Creation operators for TimeTensors
# Date : 3, August 2021
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
# Copyright Nils Schaetti <nils.schaetti@unine.ch>, <nils.schaetti@unige.ch>


# Imports
import torch
import numpy as np
import echotorch


# Create time-tensor with timetensor()
x1 = echotorch.timetensor(torch.zeros(4, 100, 6), time_dim=1)
print("x1: {}".format(x1))
print("x1.time_dim: {}".format(x1.time_dim))
print("x1.size(): {}".format(x1.size()))
print("x1.csize(): {}".format(x1.csize()))
print("x1.bsize(): {}".format(x1.bsize()))
print("x1.tlen: {}".format(x1.tlen))
print("x1.lengths: {}".format(x1.tlens))
print("")

# Create time-tensor with as_timetensor()
x2 = echotorch.as_timetensor(np.zeros((4, 100, 6)), time_dim=1)
print("x2: {}".format(x2))
print("x2.time_dim: {}".format(x2.time_dim))
print("x2.size(): {}".format(x2.size()))
print("x2.csize(): {}".format(x2.csize()))
print("x2.bsize(): {}".format(x2.bsize()))
print("x2.tlen: {}".format(x2.tlen))
print("x2.lengths: {}".format(x2.tlens))
print("")

# Create time-tensor with from_numpy()
x3 = echotorch.from_numpy(np.zeros((4, 100, 6)), time_dim=1)
print("x3: {}".format(x3))
print("x3.time_dim: {}".format(x3.time_dim))
print("x3.size(): {}".format(x3.size()))
print("x3.csize(): {}".format(x3.csize()))
print("x3.bsize(): {}".format(x3.bsize()))
print("x3.tlen: {}".format(x3.tlen))
print("x3.lengths: {}".format(x3.tlens))
print("")

# Create a time-tensor with full()
x4 = echotorch.full((6,), fill_value=5, time_length=100)
print("x4: {}".format(x4))
print("x4.time_dim: {}".format(x4.time_dim))
print("x4.size(): {}".format(x4.size()))
print("x4.csize(): {}".format(x4.csize()))
print("x4.bsize(): {}".format(x4.bsize()))
print("x4.tlen: {}".format(x4.tlen))
print("x4.lengths: {}".format(x4.tlens))
print("")

# Create a time-tensor with full() and multiple lengths
x5 = echotorch.full((6,), fill_value=5, time_length=torch.LongTensor([[100], [50]]))
print("x5: {}".format(x5))
print("x5.time_dim: {}".format(x5.time_dim))
print("x5.size(): {}".format(x5.size()))
print("x5.csize(): {}".format(x5.csize()))
print("x5.bsize(): {}".format(x5.bsize()))
print("x5.tlen: {}".format(x5.tlen))
print("x5.lengths: {}".format(x5.tlens))
print("")

# Create a time-tensor with randn()
x6 = echotorch.randn((2,), time_length=100)
print("x6: {}".format(x6))
print("x6.time_dim: {}".format(x6.time_dim))
print("x6.size(): {}".format(x6.size()))
print("x6.csize(): {}".format(x6.csize()))
print("x6.bsize(): {}".format(x6.bsize()))
print("x6.tlen: {}".format(x6.tlen))
print("x6.lengths: {}".format(x6.tlens))
print("")
