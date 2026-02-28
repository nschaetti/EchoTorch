# -*- coding: utf-8 -*-
#
# File : examples/timetensors/visualisation.py
# Description : Example of visualisation functions for timetensors.
# Date : 17th of August 2021.
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
import echotorch.viz


# Random timeseries
x = echotorch.randn(5, time_length=100)

# Pairs visualisation
echotorch.viz.pairs(x, figsize=(12, 8), s=3, sign_level=0.5)

# Difference operators
dx = echotorch.diff(x)

# Show difference
plt.figure()
echotorch.viz.timeplot(dx, title="diff(x)")
plt.show()
