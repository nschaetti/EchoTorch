# -*- coding: utf-8 -*-
#
# File : examples/conceptors/boolean_operations.py
# Description : Conceptor boolean operation
# Date : 16th of December, 2019
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
import echotorch.nn.conceptors as ecnc
import echotorch.utils.matrix_generation as mg
import echotorch.utils.visualisation as ecvs
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import math

# region PARAMS

# Init random numb.
torch.random.manual_seed(1)
np.random.seed(1)

# Type params
dtype = torch.float64

# Reservoir params
reservoir_size = 2

# endregion PARAMS

# Argument parsing
parser = argparse.ArgumentParser(prog=u"Boolean operations", description="Boolean operation demo")
parser.add_argument("--x", type=str, default="", required=False)
parser.add_argument("--x-name", type=str, default="", required=False)
parser.add_argument("--y", type=str, default="", required=False)
parser.add_argument("--y-name", type=str, default="", required=False)
args = parser.parse_args()

# region MATRICES_INIT

# Load X state matrix from file or from random distrib ?
if args.x != "":
    # Load state matrix X
    x_generator = mg.matrix_factory.get_generator(
        "matlab",
        file_name=args.x,
        entity_name=args.x_name
    )
else:
    # Generate internal weights
    x_generator = mg.matrix_factory.get_generator(
        "normal",
        mean=0.0,
        std=1.0
    )
# end if

# Load Y state matrix from file or from random distrib ?
if args.y != "":
    # Load state matrix X
    y_generator = mg.matrix_factory.get_generator(
        "matlab",
        file_name=args.y,
        entity_name=args.y_name
    )
else:
    # Generate internal weights
    y_generator = mg.matrix_factory.get_generator(
        "normal",
        mean=0.0,
        std=1.0
    )
# end if

# Generate X and Y
X = x_generator.generate(size=(reservoir_size, reservoir_size), dtype=dtype)
Y = y_generator.generate(size=(reservoir_size, reservoir_size), dtype=dtype)

# Transpose on time dim / reservoir dim
X = X.t()
Y = Y.t()

# Add batch dimension
X = X.reshape(1, reservoir_size, reservoir_size)
Y = Y.reshape(1, reservoir_size, reservoir_size)

# endregion MATRICES_INIT

# region CONCEPTOR_A

# Create a conceptor
A = ecnc.Conceptor(input_dim=reservoir_size, aperture=1, dtype=dtype)

# Learn from state
A.filter_fit(X)

# Divide correlation matrix R by reservoir dimension
# and update C.
Ra = A.correlation_matrix()
A.set_R(Ra)

# Get conceptor matrix
Ua, Sa, Va = A.SVD

# Change singular values
Sa[0] = 0.95
Sa[1] = 0.2

# New C
Cnew = torch.mm(torch.mm(Ua, torch.diag(Sa)), Va)

# Recompute conceptor
A.set_C(Cnew, aperture=1)

# endregion CONCEPTOR_A

# region CONCEPTOR_B

# Create a conceptor
B = ecnc.Conceptor(input_dim=reservoir_size, aperture=1, dtype=dtype)

# Learn from state
B.filter_fit(Y)

# Divide correlation matrix R by reservoir dimension
# and update C.
Rb = B.correlation_matrix()
B.set_R(Rb)

# Get conceptor matrix
Ub, Sb, Vb = B.SVD

# Change singular values
Sb[0] = 0.8
Sb[1] = 0.3

# Recompute conceptor
B.set_C(torch.mm(torch.mm(Ub, torch.diag(Sb)), Vb), aperture=1)

# endregion CONCEPTOR_B

# region BOOLEAN_OPERATIONS

# AND, OR, NOT
AandB = ecnc.Conceptor.operator_AND(A, B)
AorB = ecnc.Conceptor.operator_OR(A, B)
notA = ecnc.Conceptor.operator_NOT(A)
print(A.conceptor_matrix())
print(B.conceptor_matrix())
print(AandB.conceptor_matrix())
print(AorB.conceptor_matrix())
print(notA.conceptor_matrix())
# endregion BOOLEAN_OPERATIONS

# region PLOTS

# Figure
plt.figure(figsize=(12, 4))

# region PLOT_OR

# Select subplot
plt.subplot(1, 3, 1)

# Plot cross and circle
plt.plot([-1, 1], [0, 0], '--', color='black', linewidth=1)
plt.plot([0, 0], [-1, 1], '--', color='black', linewidth=1)
plt.plot(
    np.cos(2.0 * math.pi * np.arange(200) / 200.0),
    np.sin(2.0 * math.pi * np.arange(200) / 200),
    '-',
    color='black',
    linewidth=1
)

# Plot ellipse
ecvs.plot_2D_ellipse(A.conceptor_matrix(), 'red', linewidth=3, resolution=200)
ecvs.plot_2D_ellipse(B.conceptor_matrix(), 'blue', linewidth=3, resolution=200)
ecvs.plot_2D_ellipse(AorB.conceptor_matrix(), 'magenta', linewidth=6, resolution=200)

# Title
plt.title("A OR B")

# Axis sticks
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])

# endregion PLOT_OR

# region PLOT_AND

# Select subplot
plt.subplot(1, 3, 2)

# Plot cross and circle
plt.plot([-1, 1], [0, 0], '--', color='black', linewidth=1)
plt.plot([0, 0], [-1, 1], '--', color='black', linewidth=1)
plt.plot(
    np.cos(2.0 * math.pi * np.arange(200) / 200),
    np.sin(2.0 * math.pi * np.arange(200) / 200),
    '-',
    color='black',
    linewidth=1
)

# Plot ellipse
ecvs.plot_2D_ellipse(A.conceptor_matrix(), 'red', linewidth=3, resolution=200)
ecvs.plot_2D_ellipse(B.conceptor_matrix(), 'blue', linewidth=3, resolution=200)
ecvs.plot_2D_ellipse(AandB.conceptor_matrix(), 'magenta', linewidth=6, resolution=200)

# Title
plt.title("A AND B")

# Axis sticks
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])

# endregion PLOT_AND

# region PLOT_NOT

# Select subplot
plt.subplot(1, 3, 3)

# Plot cross and circle
plt.plot([-1, 1], [0, 0], '--', color='black', linewidth=1)
plt.plot([0, 0], [-1, 1], '--', color='black', linewidth=1)
plt.plot(
    np.cos(2.0 * math.pi * np.arange(200) / 200),
    np.sin(2.0 * math.pi * np.arange(200) / 200),
    '-',
    color='black',
    linewidth=1
)

# Plot ellipses
ecvs.plot_2D_ellipse(A.conceptor_matrix(), 'red', linewidth=3, resolution=200)
ecvs.plot_2D_ellipse(notA.conceptor_matrix(), 'magenta', linewidth=6, resolution=200)

# Title
plt.title("NOT A")

# Axis sticks
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])

# Show plot
plt.show()

# endregion PLOT_NOT

# endregion PLOTS
