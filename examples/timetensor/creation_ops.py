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
print("timetensor")
x1 = echotorch.timetensor(torch.zeros(4, 100, 6), time_dim=1)
# print("x1: {}".format(x1))
print("x1.time_dim: {}".format(x1.time_dim))
print("x1.size(): {}".format(x1.size()))
print("x1.csize(): {}".format(x1.csize()))
print("x1.bsize(): {}".format(x1.bsize()))
print("x1.tlen: {}".format(x1.tlen))
print("")

# Create time-tensor with as_timetensor()
print("as_timetensor")
x2 = echotorch.as_timetensor(np.zeros((4, 100, 6)), time_dim=1)
# print("x2: {}".format(x2))
print("x2.time_dim: {}".format(x2.time_dim))
print("x2.size(): {}".format(x2.size()))
print("x2.csize(): {}".format(x2.csize()))
print("x2.bsize(): {}".format(x2.bsize()))
print("x2.tlen: {}".format(x2.tlen))
print("")

# Create time-tensor with from_numpy()
print("from_numpy")
x3 = echotorch.from_numpy(np.zeros((4, 100, 6)), time_dim=1)
# print("x3: {}".format(x3))
print("x3.time_dim: {}".format(x3.time_dim))
print("x3.size(): {}".format(x3.size()))
print("x3.csize(): {}".format(x3.csize()))
print("x3.bsize(): {}".format(x3.bsize()))
print("x3.tlen: {}".format(x3.tlen))
print("")

# Create a time-tensor with full()
print("full")
x4 = echotorch.full(6, fill_value=5, length=100)
# print("x4: {}".format(x4))
print("x4.time_dim: {}".format(x4.time_dim))
print("x4.size(): {}".format(x4.size()))
print("x4.csize(): {}".format(x4.csize()))
print("x4.bsize(): {}".format(x4.bsize()))
print("x4.tlen: {}".format(x4.tlen))
print("")

# Create a time-tensor with full() and multiple lengths
print("full")
x5 = echotorch.full(6, fill_value=5, time_length=torch.LongTensor([[100], [50]]))
# print("x5: {}".format(x5))
print("x5.time_dim: {}".format(x5.time_dim))
print("x5.size(): {}".format(x5.size()))
print("x5.csize(): {}".format(x5.csize()))
print("x5.bsize(): {}".format(x5.bsize()))
print("x5.tlen: {}".format(x5.tlen))
print("")

# Create a time-tensor with randn()
print("randn")
x6 = echotorch.randn(2, time_length=100)
# print("x6: {}".format(x6))
print("x6.time_dim: {}".format(x6.time_dim))
print("x6.size(): {}".format(x6.size()))
print("x6.csize(): {}".format(x6.csize()))
print("x6.bsize(): {}".format(x6.bsize()))
print("x6.tlen: {}".format(x6.tlen))
print("")

# Create a sparse COO timetensor
print("sparse_coo_timetensor")
x7 = echotorch.sparse_coo_timetensor(
    indices=torch.tensor([[0, 1, 1], [2, 0, 2]]),
    values=torch.tensor([3, 4, 5], dtype=torch.float32),
    size=[2, 4]
)
print("x7: {}".format(x7))
print("x7.time_dim: {}".format(x7.time_dim))
print("x7.size(): {}".format(x7.size()))
print("x7.csize(): {}".format(x7.csize()))
print("x7.bsize(): {}".format(x7.bsize()))
print("x7.tlen: {}".format(x7.tlen))
print("")

# As strided
# print("as_strided")
# x8 = echotorch.as_strided(x6, )

# Create timetensor full of zeros
print("zeros")
x9 = echotorch.zeros(2, time_length=100)
print("x9.time_dim: {}".format(x9.time_dim))
print("x9.size(): {}".format(x9.size()))
print("x9.csize(): {}".format(x9.csize()))
print("x9.bsize(): {}".format(x9.bsize()))
print("x9.tlen: {}".format(x9.tlen))
print("")

# Create timetensor with arange
x10 = echotorch.arange(1, 2.5, 0.5)
print("x10: {}".format(x10))
print("x10.time_dim: {}".format(x10.time_dim))
print("x10.size(): {}".format(x10.size()))
print("x10.csize(): {}".format(x10.csize()))
print("x10.bsize(): {}".format(x10.bsize()))
print("x10.tlen: {}".format(x10.tlen))
print("")

# Create timetensor with linspace
x11 = echotorch.linspace(-10, 10, steps=1)
print("x11: {}".format(x11))
print("x11.time_dim: {}".format(x11.time_dim))
print("x11.size(): {}".format(x11.size()))
print("x11.csize(): {}".format(x11.csize()))
print("x11.bsize(): {}".format(x11.bsize()))
print("x11.tlen: {}".format(x11.tlen))
print("")

# Create timetensor with logspace
x12 = echotorch.logspace(start=2, end=2, steps=1, base=2)
print("x12: {}".format(x12))
print("x12.time_dim: {}".format(x12.time_dim))
print("x12.size(): {}".format(x12.size()))
print("x12.csize(): {}".format(x12.csize()))
print("x12.bsize(): {}".format(x12.bsize()))
print("x12.tlen: {}".format(x12.tlen))
print("")

# Create timetensor with empty
x13 = echotorch.empty(2, time_length=100)
print("x13.time_dim: {}".format(x13.time_dim))
print("x13.size(): {}".format(x13.size()))
print("x13.csize(): {}".format(x13.csize()))
print("x13.bsize(): {}".format(x13.bsize()))
print("x13.tlen: {}".format(x13.tlen))
print("")

# Create timetensor with empty_like
x13 = echotorch.empty_like(x13)
print("x13.time_dim: {}".format(x13.time_dim))
print("x13.size(): {}".format(x13.size()))
print("x13.csize(): {}".format(x13.csize()))
print("x13.bsize(): {}".format(x13.bsize()))
print("x13.tlen: {}".format(x13.tlen))
print("")


# Create timetensor with empty_strided
x14 = echotorch.empty_strided((2, 3), (1, 2), time_length=100, time_stride=2)
print("x14.time_dim: {}".format(x14.time_dim))
print("x14.size(): {}".format(x14.size()))
print("x14.csize(): {}".format(x14.csize()))
print("x14.bsize(): {}".format(x14.bsize()))
print("x14.tlen: {}".format(x14.tlen))
print("")


