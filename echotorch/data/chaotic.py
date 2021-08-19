# -*- coding: utf-8 -*-
#
# File : echotorch/datasets/functional/chaotic.py
# Description : Attractor and chaos-based timeseries generation.
# Date : 10th of August, 2021
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
from typing import Union, Tuple, List
import torch
import echotorch
from random import shuffle


# Henon attractor
def henon(
        size: int,
        length: int,
        xy: Tuple[float, float],
        a: float = 1.4,
        b: float = 0.3,
        washout: int = 0
) -> Tuple[echotorch.TimeTensor]:
    """Generate a series with the Hénon map dynamical system.

    Definition
        From Wikipedia: The Hénon map, sometimes called **Hénon-Pomeau attractor/map** is a discrete-time dynamical system.
        It is one of the most studied examples of dynamical systems that exhibit chaotic behavior.
        The Hénon map takes a point :math:`(x_n, y_n)` in the plane is mapped to the new point

        .. math::
            :nowrap:

            \[
            \\begin{cases}
            x_{n+1} = 1 - a x_n^2 + y_n \\\\
            y_{n+1} = b x_n
            \\end{cases}
            \]

        The map depends on two parameters, **a** and **b**, which for the **classical Hénon map** have values of a = 1.4 and
        b = 0.3. For the classical values the Hénon map is chaotic. For other values of a and b the map may be
        chaotic, intermittent, or converge to a periodic orbit.

    :param size: How many samples to generate
    :type size: ``int``
    :param length: Length of samples (time)
    :type length: ``int``
    :param xy: Starting position in the xy-plane
    :type xy: Tuple of ints
    :param a: System parameter (default: 1.4)
    :type a: Float
    :param b: Secodn system parameter (default: 0.3)
    :type b: Float
    :param washout: Time steps to remove at the beginning of samples
    :type washout: int (default: 0)
    :return: A ``list`` of ``TimeTensor`` with series generated from Henon's equations
    :rtype: ``tuple`` of ``TimeTensor``

    Example
        >>> x = echotorch.datasets.functional.henon(1, 100, xy=(0, 0), a=1.4, b=0.3)
        >>> x
        timetensor(tensor([[ 1.0000,  0.0000],
                           [-0.4000,  0.3000],
                           [ 1.0760, -0.1200],
                           [-0.7409,  0.3228],
                           [ 0.5543, -0.2223],
                           ...
                           [ 0.9608,  0.1202],
                           [-0.1721,  0.2882],
                           [ 1.2468, -0.0516],
                           [-1.2279,  0.3740]]), time_dim: 0, tlen: 100)
        >>> echotorch.utils.timepoints2d(x)
    """
    # Samples
    samples = list()

    # Henon functions
    def henon_func(x: float, y: float) -> torch.Tensor:
        x_dot = 1 - (a * (x * x)) + y
        y_dot = b * x
        return torch.Tensor([x_dot, y_dot])
    # end henon_func

    # Washout
    for t in range(washout):
        xy = henon_func(xy[0], xy[1])
    # end for

    # For each sample
    for n in range(size):
        # Tensor
        sample = echotorch.zeros((2,), time_length=length)

        # Timesteps
        for t in range(length):
            xy = henon_func(xy[0], xy[1])
            sample[t] = xy
        # end for

        # Add
        samples.append(sample)
    # end for

    # Shuffle
    shuffle(samples)

    return samples
# end henon
