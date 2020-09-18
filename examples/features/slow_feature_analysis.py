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
import sksfa
import echotorch.nn as etnn


# Time length
time_length = 2.0 * math.pi

# Time step
time_step = 0.0001

# Number of components to extract
n_components = 2

# Total number of samples in the timeseries
total_points = time_length / time_step

# Length to plot
plot_length = 2.0 * math.pi

# Number of points to plot
plot_points = int(plot_length / time_step)

# Time values
ts_gen = torch.arange(0, time_length, time_step)
ts = [t for t in ts_gen]

# Generate the first signal
# x1(t) = sin(t) + cos(11t)^2
x1 = torch.tensor([[torch.sin(t) + torch.pow(torch.cos(11.0 * t), 2)] for t in ts_gen], dtype=torch.float64)

# Generate the second signal
# x2(t) = cos(11*t)
x2 = torch.tensor([[torch.cos(11.0 * t)] for t in ts_gen], dtype=torch.float64)

# Compute x
x = torch.cat((x1, x2), dim=1)
x = x - torch.mean(x, dim=0)
x = x / torch.std(x, dim=0)

# Plot the first signal
sin_x = torch.tensor([[torch.sin(t)] for t in ts_gen])
plt.plot(ts[:plot_points], x[:plot_points, 0].numpy(), color='r')
plt.plot(ts[:plot_points], sin_x[:plot_points, 0], color='black')
plt.xticks([0, math.pi, 2.0*math.pi])
plt.yticks([-1, 0, 1])
plt.ylim(-2, 2)
plt.title("Input component x1(t)")
plt.show()

# Plot the second signal
plt.plot(ts[:plot_points], x[:plot_points, 1].numpy(), color='r')
plt.xticks([0, math.pi, 2.0*math.pi])
plt.yticks([-1, 0, 1])
plt.ylim(-2, 2)
plt.title("Input component x2(t)")
plt.show()

# Plot in 2D space
plt.plot(x[:plot_points, 1].numpy(), x[:plot_points, 0].numpy(), color='r')
plt.xticks([-1, 0, 1])
plt.yticks([-2, -1, 0, 1, 2])
plt.xlabel("x2(t)")
plt.ylabel("x1(t)")
plt.title("Input trajectory x(t)")
plt.show()


#
# EXPANDED SIGNAL
#


# Compute the expended signal z(t)
def h(u):
    return torch.cat(
        (
            u,
            torch.mul(u[:, 1], u[:, 1]).reshape(-1, 1),
            torch.mul(u[:, 0], u[:, 0]).reshape(-1, 1),
            torch.mul(u[:, 0], u[:, 1]).reshape(-1, 1)
        ),
        dim=1
    )
    # return u
# end h


# Compute expanded signal z
z = h(x)

# Plot expanded signal in 3D space
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot3D(z[:plot_points, 2], z[:plot_points, 1], z[:plot_points, 0], 'r')
ax.set_xlabel("z5")
ax.set_xlim(-1, 2)
ax.set_ylabel("z2")
ax.set_ylim(-1, 1)
ax.set_zlabel("z1")
ax.set_zlim(-2, 2)
plt.title("Expanded signal z(t)")
plt.show()


#
# SFA
#

# SFA node
sfa_node = etnn.SFACell(
    input_dim=5,
    output_dim=n_components,
    time_step=time_step,
    dtype=torch.float64
)

# Add a batch dimension to z
z = z.unsqueeze(0)

# Training
y = sfa_node(z)
print(y.size())
# Show components from home made SFA
plt.title("Components from SFA cell")
plt.plot(ts[:plot_points], y[0, :plot_points, 0].numpy(), color='r')
plt.plot(ts[:plot_points], y[0, :plot_points, 1].numpy(), color='b')
plt.xticks([0, math.pi, 2.0*math.pi])
plt.yticks([-1, 0, 1])
plt.ylim(-2, 2)
plt.show()
