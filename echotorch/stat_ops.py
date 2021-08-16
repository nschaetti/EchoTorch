# -*- coding: utf-8 -*-
#
# File : echotorch/stat_ops.py
# Description : Statistical operations on (Time/Data/*)Tensor-
# Date : 16th of August, 2021
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
from torch import Tensor, mean, mm

# Import local
from .timetensors import TimeTensor


# Average over time dimension
def tmean(
        input: TimeTensor
) -> Tensor:
    r"""Returns the mean value over time dimension of all elements in the ``input`` timetensor.

    :param input: the input timetensor.
    :type input: ``TimeTensor``
    """
    return mean(input, dim=input.time_dim)
# end tmean


# Covariance matrix
def cov(
        t1: TimeTensor,
        t2: TimeTensor
) -> Tensor:
    r"""Returns the covariance matrix of two 1-D timeseries with the same number of channels.

    :param t1: first timetensor.
    :type t1: ``TimeTensor``
    :param t2: second timetensor.
    :type t2: ``TimeTensor``
    :return: The covariance matrix of the two timeseries computed over the time dimension.
    :rtype: ``Tensor``

    Example:
        >>> x = echotorch.randn(5, time_length=100)
        >>> y = echotorch.randn(5, time_length=100)
        >>> echotorch.cov(x, y)
        tensor([[-0.1720,  0.1031,  0.1089,  0.0310, -0.1076],
            [-0.1715, -0.0077,  0.0133,  0.1061,  0.0222],
            [ 0.0762, -0.0763, -0.0502, -0.0925, -0.0855],
            [-0.0147, -0.0295, -0.1638, -0.0070, -0.1023],
            [ 0.1776, -0.2714,  0.0152, -0.0449,  0.0721]])
    """
    # Check that t1 and t2 have the time dim at pos 0
    if t1.time_dim != 0 or t2.time_dim != 0:
        raise ValueError(
            "Expected two timeseries with time dimension first (here {} and {}".format(t1.time_dim, t2.time_dim)
        )
    # end if

    # Check that t1 and t2 have the same time length
    if t1.tlen != t2.tlen:
        raise ValueError(
            "Expected two timeseries with same time lengths (here {} != {})".format(t1.tlen, t2.tlen)
        )
    # end if

    # Compute means
    t1_mean = tmean(t1)
    t2_mean = tmean(t2)

    # Compute covariance
    return mm((t1 - t1_mean).t(), t2 - t2_mean) / t1.tlen
# end cov

