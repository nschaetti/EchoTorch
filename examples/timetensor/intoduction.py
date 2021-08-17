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
import numpy as np
import echotorch


# Create a timetensor
x = echotorch.timetensor([1, 2, 3], time_dim=0)
# print(x)

# Zeros
x = echotorch.zeros(4, 4, time_length=20)
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
x = echotorch.rand(4, 4, time_length=20).cpu()
y = echotorch.rand(4, 4, time_length=30).cpu()
print("Base timetenors: ")
print("x size: {}".format(x.size()))
print("x time_dim: {}".format(x.time_dim))
print("x time_length: {}".format(len(x)))
print("x csize: {}".format(x.csize()))
print("")
print("y size: {}".format(y.size()))
print("y time_dim: {}".format(y.time_dim))
print("y time_length: {}".format(len(y)))
print("y csize: {}".format(y.csize()))
print("")

# Equal
print("==")
print("x == y: {}".format(x == y))
print("x == x: {}".format(x == x))
print("")

# Cat
xy = torch.cat((x, y), dim=0)
print("torch.cat dim=0:")
print("out time_dim: {}".format(xy.time_dim))
print("out time_length: {}".format(len(xy)))
print("out csize: {}".format(xy.csize()))
print("")

# Unsqueeze
xx = torch.unsqueeze(x, dim=0)
print("torch.unsqueeze")
print("out size: {}".format(xx.size()))
print("out time_length: {}".format(len(xx)))
print("out time dim: {}".format(xx.time_dim))
print("")

# tmean
z = echotorch.randn(5, time_length=100)
z2 = echotorch.randn(5, time_length=100)
print("echotorch.tmean")
print("z size: {}".format(z.size()))
print("z tmean: {}".format(echotorch.tmean(z)))
print("z z1 cov: {}".format(echotorch.cov(z, z2)))
print("z z cov: {}".format(echotorch.cov(z, z)))
z_n = z.numpy()
z2_n = z2.numpy()
print("z_n: {}".format(z_n.shape))
z_n_cov = np.cov(z_n, rowvar=False)
print("z_n_cov: {}".format(z_n_cov.shape))
print("z z cov with numpy: {}".format(z_n_cov))
z_z2_n_cov = np.cov(z_n, z2_n, rowvar=False)
print("z z2 cov with numpy: {}".format(z_z2_n_cov[:5, 5:]))
