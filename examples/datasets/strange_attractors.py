# -*- coding: utf-8 -*-
#
# File : examples/datasets/strange_attractors.py
# Description : Examples of time series generation based on chaotic and strange attractors
# Date : 12th of August, 2021
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
import matplotlib.pyplot as plt
import echotorch.data
import echotorch.viz


# Henon strange attractor
henon_series = echotorch.data.henon(
    size=1,
    length=100,
    xy=(0, 0),
    a=1.4,
    b=0.3,
    washout=0
)

# Show points
plt.figure()
echotorch.viz.timescatter(henon_series[0], title="Henon Attractor", xlab="Feature 1", ylab="Feature 2")
plt.show()
