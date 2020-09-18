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
x1 = torch.tensor([[torch.sin(t) + torch.pow(torch.cos(11.0 * t), 2)] for t in ts_gen])

# Generate the second signal
# x2(t) = cos(11*t)
x2 = torch.tensor([[torch.cos(11.0 * t)] for t in ts_gen])

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
# WHITENING SIGNAL
#

# Compute the bias
b = -torch.mean(z, dim=0)

# Centered Z
zm = z + b

# Compute the covariance matrix of z
cov_zm = torch.mm(zm.t(), zm) / total_points

# Show the covariance matrix
plt.imshow(cov_zm.numpy(), cmap='Greys')
plt.title("Centerez z covariance matrix")
plt.show()

# Compute the eigenvectors and eigenvalues of the
# covariance of zm
D, U = torch.eig(cov_zm, eigenvectors=True)

# Remove imaginary part and compute the diagonal matrix
D = torch.diag(D[:, 0])

# Compute S,  the linear transformation to normalize the signal
# S = L^-1/2 * Q^T
S = torch.mm(torch.sqrt(torch.inverse(D)), U.t())

# Normalize the expanded signal z with the linear transformation
# zs = sqrt(L^-1) * Q^-1 * zm
zs = torch.mm(S, zm.t()).t()

# Print average of zs to checked
# centeredness
print("Average of zs : {}".format(torch.mean(zs, dim=0)))

# Compute the covariance matrix of zs
cov_zsT = torch.mm(zs.t(), zs) / total_points

# Show the new covariance matrix
plt.imshow(cov_zsT.numpy(), cmap='Greys')
plt.title("Sphered z covariance matrix")
plt.show()

# Plot sphered expanded signal in 3D space
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot3D(zs[:plot_points, 2], zs[:plot_points, 1], zs[:plot_points, 0], 'r')
ax.set_xlabel("zs5")
ax.set_xlim(-1, 2)
ax.set_ylabel("zs2")
ax.set_ylim(-1, 1)
ax.set_zlabel("zs1")
ax.set_zlim(-2, 2)
plt.title("Sphered expanded signal zs(t)")
plt.show()


#
# TIME DERIVATIVES
#


# Compute the time derivative of zs
dzs = (zs[1:] - zs[:-1]) / time_step

# Covariance matrix of dzs
cov_dzs = torch.mm(dzs.t(), dzs) / total_points

# Compute eignen decomposition on time derivative
L, V = torch.eig(cov_dzs, eigenvectors=True)
print(L)
# Keep only the needed components
# V = V[:, :n_components]

# Compute W
W = torch.mm(V.t(), S)


def g(u):
    # To expanded form
    hu = h(u)

    # In component form
    return torch.mm(W, (hu + b).t()).t()
# end g_func


# Invariant features
sf = g(x)

sfa_transformer = sksfa.SFA(n_components=2, )
sf_x = sfa_transformer.fit_transform(zm.numpy())

# Plot derivative and expanded signal, in 3D space
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot3D(zs[:plot_points, 2]*10.0, zs[:plot_points, 1]*10.0, zs[:plot_points, 0]*10.0, 'r')
ax.plot3D(dzs[:plot_points, 2], dzs[:plot_points, 1], dzs[:plot_points, 0], 'b')
ax.set_xlabel("dx1x2")
ax.set_xlim(-25, 25)
ax.set_ylabel("dx2")
ax.set_ylim(-25, 25)
ax.set_zlabel("dx1")
ax.set_zlim(-25, 25)
plt.show()

# Show components from home made SFA
plt.title("Components from home made SFA")
plt.plot(ts[:plot_points], sf[:plot_points, 0].numpy(), color=(1.0, 0.0, 0.0, 0.5))
plt.plot(ts[:plot_points], sf[:plot_points, 1].numpy(), color='g')
plt.plot(ts[:plot_points], sf[:plot_points, 2].numpy(), color=(0.0, 0.0, 1.0, 0.5))
plt.plot(ts[:plot_points], sf[:plot_points, 3].numpy(), 'black')
plt.plot(ts[:plot_points], sf[:plot_points, 4].numpy(), color=(1.0, 1.0, 0.0, 0.5))
plt.xticks([0, math.pi, 2.0*math.pi])
plt.yticks([-1, 0, 1])
plt.ylim(-2, 2)
plt.show()

# Show components from sklearn
plt.title("Components from sklearn SFA")
plt.plot(ts[:plot_points], sf_x[:plot_points, 0], 'r')
plt.plot(ts[:plot_points], sf_x[:plot_points, 1], 'g')
plt.xticks([0, math.pi, 2.0*math.pi])
plt.yticks([-1, 0, 1])
plt.ylim(-2, 2)
plt.show()
