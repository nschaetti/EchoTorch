# -*- coding: utf-8 -*-
#
# File : echotorch/examples/features/slow_feature_analysis.py
# Description : Example of the Slow Feature Analysis principle.
# Date : 16th of September, 2020
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
import math
import torch
import matplotlib.pyplot as plt


# Time length
time_length = 100
plot_length = 2.0 * math.pi
plot_points = 630

# Time values
ts_gen = torch.arange(0, time_length, 0.01)
ts = [t for t in ts_gen]

# Generate the first signal
# x1(t) = sin(t) + cos(11t)^2
x1 = torch.tensor([[torch.sin(t) + torch.pow(torch.cos(11.0*t), 2)] for t in ts_gen])

# Generate the second signal
# x2(t) = cos(11*t)
x2 = torch.tensor([[torch.cos(11.0*t)] for t in ts_gen])

# Compute the expended signal z(t)
z = torch.cat(
    (
        x1,
        x2,
        torch.mul(x1, x2),
        torch.mul(x1, x1),
        torch.mul(x2, x2)
    ),
    dim=1
)

# Sphering the expanded signal
# PCA on (z(t) - <z(t)>)
zm = z - torch.mean(z, dim=0)
zmTzm = torch.mm(zm.t(), zm)
U, S, _ = torch.svd(zmTzm)
zs = torch.mm(U, zm.t()).t()
print("Average of zs : {}".format(torch.mean(zs, dim=0)))
average_zzT = torch.zeros(5, 5)
for t in range(time_length):
    average_zzT += torch.mm(zs[t].reshape(5, 1), zs[t].reshape(1, 5))
# end for
average_zzT /= time_length
print("average_zzT : {}".format(average_zzT))
plt.imshow(average_zzT.numpy(), cmap='Greys')
plt.show()

# Plot the first signal
plt.plot(ts[:plot_points], x1[:plot_points, 0].numpy(), color='r')
plt.xticks([0, math.pi, 2.0*math.pi])
plt.yticks([-1, 0, 1])
plt.title("Input component x1(t)")
plt.show()

# Plot the second signal
plt.plot(ts[:plot_points], x2[:plot_points, 0].numpy(), color='r')
plt.xticks([0, math.pi, 2.0*math.pi])
plt.yticks([-1, 0, 1])
plt.title("Input component x2(t)")
plt.show()

# Plot in 2D space
plt.plot(x2[:plot_points, 0].numpy(), x1[:plot_points, 0].numpy(), color='r')
plt.xticks([-1, 0, 1])
plt.yticks([-2, -1, 0, 1, 2])
plt.title("Input trajectory x(t)")
plt.show()

# Plot expanded signal in 3D space
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot3D(z[:plot_points, -1], z[:plot_points, 1], z[:plot_points, 0], 'r')
ax.set_xlabel("x1x2")
ax.set_xlim(0, 2)
ax.set_ylabel("x2")
ax.set_ylim(-1, 1)
ax.set_zlabel("x1")
ax.set_zlim(-2, 2)
plt.show()

# Plot sphered expanded signal in 3D space
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot3D(zs[:plot_points, -1], zs[:plot_points, 1], zs[:plot_points, 0], 'r')
ax.set_xlabel("x1x2")
ax.set_xlim(0, 2)
ax.set_ylabel("x2")
ax.set_ylim(-1, 1)
ax.set_zlabel("x1")
ax.set_zlim(-2, 2)
plt.show()
