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


# Create a two timetensors
# x = echotorch.rand(5, length=100)
# y = echotorch.rand(5, length=100)
x = echotorch.data.ma(1, length=1000, order=5, size=1)[0]
y = echotorch.data.ma(1, length=1000, order=5, size=1)[0]

# Print tensors
print("Timetensor x: {}".format(x))
print("Timetensor y: {}".format(y))
print("")

# Average mean over time
xtm = echotorch.tmean(x)
ytm = echotorch.tmean(y)

# Show t-mean
print("Average over time of x: {}".format(xtm))
print("Average over time of y: {}".format(ytm))
print("")

# Standard deviation over time
xsig = echotorch.tstd(x)
ysig = echotorch.tstd(y)

# Show standard deviation over time
print("Std over time of x: {}".format(xsig))
print("Std over time of y: {}".format(ysig))
print("")

# Variance over time
xvar = echotorch.tvar(x)
yvar = echotorch.tvar(y)

# Show variance over time
print("Var over time of x: {}".format(xvar))
print("Var over time of y: {}".format(yvar))
print("")

# Compute covariance matrix
cov_xy = echotorch.cov(x, y)

# Show covariance matrix
print("Cov(X, Y): {}".format(cov_xy))
print("")

# Compute correlation matrix
cor_xy = echotorch.cor(x, y, pvalue=True)

# Show correlation matrix
print("Cor(X, Y): {}".format(cor_xy))
print("Cor(X, X): {}".format(echotorch.cor(x, x)))
print("")

# Compute auto-covariance coefficients
autocov_coeffs = echotorch.acf.acf(x, k=50)

# Show autocov coeffs
echotorch.acf.correlogram(x, k=50, plot_params={'title': "Auto-covariance coefficients"})

# Compute auto-correlation coefficients
echotorch.acf.correlogram(x, k=50, coeffs_type="correlation", plot_params={'title': "Auto-correlation coefficients"})

# Compute cross auto-correlation coefficients
echotorch.acf.cross_correlogram(x, y, k=50, coeffs_type="correlation", plot_params={'title': "Cross Autocorrelation coefficients"})
