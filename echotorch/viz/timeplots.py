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
from typing import List, Any
import matplotlib.pyplot as plt

# Import echotorch
from echotorch import TimeTensor


# Show a 2D-timeseries as a set of points
def timescatter(
        data: TimeTensor,
        create_figure: bool = True,
        show_figure: bool = True,
        figure_options: List[Any] = None,
        plot_options: List[Any] = None
) -> None:
    r"""Show a 2D-timeseries on a 2D-plane as a set of points

    :param data:
    :type data:
    :param create_figure:
    :type create_figure:
    :param show_figure:
    :type show_figure:
    :param figure_options:
    :type figure_options:
    :param plot_options:
    :type plot_options:

    Example
        >>> echotorch.timepoints2d(...)

    """
    # Create figure?
    if create_figure and figure_options is not None:
        plt.figure(**figure_options)
    elif create_figure:
        plt.figure()
    # end if

    # Plot
    if plot_options is not None:
        plt.scatter(data[:, 0], data[:, 1], **plot_options)
    else:
        plt.scatter(data[:, 0], data[:, 1])
    # end if

    # Show
    if show_figure:
        plt.show()
    # end if
# end timescatter

