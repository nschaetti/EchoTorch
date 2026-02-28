# -*- coding: utf-8 -*-
#
# File : examples/timetensor/statistics.py
# Description : Statistical operations on TimeTensors
# Date : 17th of August, 2021
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
import numpy as np
import matplotlib.pyplot as plt
import echotorch.data
import echotorch.acf
import echotorch.viz


# Three 1-D timeseries from Moving Average of order 5 MA(5)
x = echotorch.data.ma(1, length=1000, order=5, size=1)[0]
y = echotorch.data.ma(1, length=1000, order=5, size=1)[0]
z = echotorch.data.ma(1, length=1000, order=5, size=5)[0]

# Compute auto-covariance coefficients
autocov_coeffs = echotorch.acf.acf(x, k=50)

# Show autocov coeffs
echotorch.acf.correlogram(x, k=50, plot_params={'title': "Auto-covariance coefficients"})

# Compute auto-correlation coefficients
echotorch.acf.correlogram(x, k=50, coeffs_type="correlation", plot_params={'title': "Auto-correlation coefficients"})

# Compute cross auto-correlation coefficients
echotorch.acf.cross_correlogram(x, y, k=50, coeffs_type="correlation", plot_params={'title': "Cross Autocorrelation coefficients"})

# Show cross-correlogram
echotorch.acf.ccfpairs(
    z,
    k=20,
    coeffs_type="correlation",
    figsize=(12, 10),
    labels=['A', 'B', 'C', 'D', 'E']
)
