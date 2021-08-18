# -*- coding: utf-8 -*-
#
# File : echotorch/utils/visualisation/visualistion.py
# Description : Visualisation utility functions
# Date : 6th of December, 2019
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
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Import echotorch
from echotorch import TimeTensor


# Show a 2D-timeseries as a set of points
def timescatter(
        data: TimeTensor,
        title: Optional[str] = None,
        xlab: Optional[str] = None,
        ylab: Optional[str] = None,
        xticks: Optional[List[float]] = None,
        yticks: Optional[List[float]] = None,
        xlim: Optional[Tuple[float]] = None,
        ylim: Optional[Tuple[float]] = None,
        **kwargs
) -> None:
    r"""Show a 2D-timeseries as a set of points on a 2D scatter plot.

    :param data: The ``TimeTensor`` to plot, there should be no batch dimensions and 2 channel dimensions.
    :type data: ``TimeTensor`` of size (time length x 2)
    :param title: Plot title
    :type title: ``str``
    :param xlab: X-axis label
    :type xlab: ``str``
    :param ylab: Y-axis label
    :type ylab: ``str``
    :param xticks: X-axis ticks
    :type xticks: List of ``float``
    :param yticks: Y-axis ticks
    :type yticks: List of ``float``
    :param xlim: X-axis start and end
    :type xlim: Tuple of ``float``
    :param ylim: Y-axis start and end
    :type ylim: Tuple of ``float``

    Example
        >>> x = echotorch.data.henon(1, 100, (0, 0), 1.4, 0.3, 0)
        >>> echotorch.timescatter(x[0], title="Henon Attractor", xlab="x", ylab="y")

    """
    # Plot properties
    if title is not None: plt.title(title)
    if xlab is not None: plt.xlabel(xlab)
    if ylab is not None: plt.ylabel(ylab)
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    if xticks is not None: plt.xticks(xticks)
    if yticks is not None: plt.yticks(yticks)

    # Plot
    plt.scatter(data[:, 0], data[:, 1], **kwargs)
# end timescatter


# Plot a timetensor
def timeplot(
        data: TimeTensor,
        title: Optional[str] = None,
        tstart: Optional[float] = 0.0,
        tstep: Optional[float] = 1.0,
        tlab: Optional[str] = "Time",
        xlab: Optional[str] = None,
        tticks: Optional[List[float]] = None,
        xticks: Optional[List[float]] = None,
        tlim: Optional[Tuple[float]] = None,
        xlim: Optional[Tuple[float]] = None,
        **kwargs
) -> None:
    r"""Show a 0-D or 1-D timeseries, one line per channel, on a plot with time as the X-axis.

    :param data: The ``TimeTensor`` to plot, there must be no batch dimensions.
    :type data: ``TimeTensor``
    :param title: Plot title
    :type title: ``str``, optional
    :param tstart: Starting time position on the Time-axis
    :type tstart: ``float``, optional
    :param tstep: Time step on the Time-axis
    :type tstep: ``float``, optional
    :param tlab: Time-axis label
    :type tlab: ``str``, optional
    :param xlab: X-axis label
    :type xlab: ``str``, optional
    :param tticks: Time-axis ticks
    :type tticks: List of ``float``, optional
    :param xticks: X-axis ticks
    :type xticks: List of ``float``, optional
    :param tlim: Time-axis start and end
    :type tlim: Tuple of ``float``, optional
    :param xlim: X-axis start and end
    :type xlim: Tuple of ``float``, optional

    Example
        >>> x = echotorch.data.random_walk(1, length=10000, shape=())
        >>> echotorch.timeplot(x[0], title="Random Walk", xlab="X_t")
    """
    # Plot properties
    if title is not None: plt.title(title)
    if tlab is not None: plt.xlabel(tlab)
    if xlab is not None: plt.ylabel(xlab)
    if tlim is not None: plt.xlim(tlim)
    if xlim is not None: plt.ylim(xlim)
    if tticks is not None: plt.xticks(tticks)
    if xticks is not None: plt.yticks(xticks)

    # 0-D or 1-D
    multi_dim = data.cdim > 0

    # Number of channels
    n_chan = data.csize()[0] if multi_dim else 0

    # X-axis
    if multi_dim:
        x_data = np.expand_dims(np.arange(tstart, tstep * data.tlen, tstep), axis=1)
        x_data = np.repeat(x_data, n_chan, axis=1)
    else:
        x_data = np.arange(tstart, tstep * data.tlen, tstep)
    # end if

    # Plot
    plt.plot(x_data, data.numpy(), **kwargs)
# end timeplot

