# -*- coding: utf-8 -*-
#
# File : examples/timetensor/introduction.py
# Description : Introduction to time tensors
# Date : 31th of July, 2021
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
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>
# University of Geneva <nils.schaetti@unige.ch>


# Imports
import torch

import echotorch


# Create a timetensor
x = echotorch.timetensor([1, 2, 3], time_dim=0)
# print(x)

# Zeros
x = echotorch.zeros((4, 4), time_length=20)
# print(x)
# print(x.size())
# print(x.tsize())
# print(x.tlen)

# Indexing
# subx = x[0, :]
# print(subx)
# print(type(subx))
# print(subx.size())
# if type(subx) is echotorch.TimeTensor: print(subx.tsize())

# # Ones
# x = echotorch.ones((4, 4), 20)
# print(x)
#
# # Empty
# x = echotorch.empty((4, 4), 20)
# print(x)
#
# # Full
# x = echotorch.full((4, 4), 20, -1)
# print(x)

# Rand
x = echotorch.rand((4, 4), 20).cpu()
y = echotorch.rand((4, 4), 30).cpu()
print("x size: {}".format(x.size()))
print("x time_dim: {}".format(x.time_dim))
print("")
print("y size: {}".format(y.size()))
print("y time_dim: {}".format(y.time_dim))
print("")

# Cat
xy = torch.cat((x, y), dim=0)
print("xy size: {}".format(xy.size()))

# Unsqueeze
xx = torch.unsqueeze(x, dim=0)
print("xx size: {}".format(xx.size()))
print("xx time dim: {}".format(xx.time_dim))

