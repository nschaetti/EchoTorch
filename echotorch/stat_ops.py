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
from typing import Optional
import torch
from torch import Tensor, mean, mm, std, var

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


# Standard deviation over time dimension
def tstd(
        input: TimeTensor,
        unbiased: bool = True
) -> Tensor:
    r"""Returns the standard deviation over time dimension of all elements in the ``input`` timetensor.

    :param input: the input timetensor.
    :type input: ``TimeTensor``
    :param unbiased: whether to used Bessel's correction (:math:`\delta N = 1`)
    :type unbiased: bool

    Example:

        >>> x = echotorch.rand(5, time_length=10)
        >>> echotorch.tstd(x)
        tensor([0.2756, 0.2197, 0.2963, 0.2962, 0.2853])
        """
    return std(input, dim=input.time_dim, unbiased=unbiased)
# end tstd


# Variance over time dimension
def tvar(
        input: TimeTensor,
        unbiased: bool = True
) -> Tensor:
    r"""Returns the variance over time dimension of all elements in the ``input`` timetensor.

    :param input: the input timetensor.
    :type input: ``TimeTensor``
    :param unbiased: whether to used Bessel's correction (:math:`\delta N = 1`)
    :type unbiased: bool

    Example:

        >>> x = echotorch.rand(5, time_length=10)
        >>> echotorch.tvar(x)
        tensor([0.0726, 0.0542, 0.0754, 0.0667, 0.0675])

    """
    return var(input, dim=input.time_dim, unbiased=unbiased)
# end tvar


# Correlation matrix
def cor(
        t1: TimeTensor,
        t2: Optional[TimeTensor] = None,
        bias: Optional[bool] = False,
        ddof: Optional[int] = None
) -> Tensor:
    r"""Returns the correlation matrix between two 1-D timeseries, ``x`` and ``y``, with the same number of channels.

    As the size of the two timetensors is :math:`(T, p)`, the returned
    matrix :math:`R` has a size :math:`(p, p)`. Each element :math:`R_{ij}` of
    matrix :math:`R` is the correlation :math:`Cor(x_i, y_i)` between :math:`x_i` and :math:`y_i`,
    such that,

    .. math::
        :nowrap:

        $$
        R =
        \begin{pmatrix}
        Cor(x_{1}, y_{1}) & Cor(x_{1}, y_{2}) & \cdots & Cor(x_{1}, y_{p}) \\
        Cor(x_{2}, y_{1}) & \ddots & \cdots & \vdots \\
        \vdots & \vdots & \ddots & \vdots \\
        Cor(x_{p}, y_{1}) & \cdots & \cdots & Cor(x_{p}, y_{p})
        \end{pmatrix}
        $$

    where :math:`p` is the number of channels.

    :param t1: first timetensor containing the uni or multivariate timeseries. The time dimension should be at position 0.
    :type t1: ``TimeTensor``
    :param t2: An additional ``TimeTensor`` with same shape and time length. If ``None``, the auto-correlation of *t1* is returned.
    :type t2: ``TimeTensor``, optional
    :param bias: Default normalization (False) is by :math:`(N - 1)`, where :math:`N` is the number of observations given (unbiased) or length of the timeseries. If *bias* is True, then normalization is by :math:`N`. These values can be overriden by using the keyword *ddof*.
    :type bias: ``bool``, optional
    :param ddof: If not *None* the default value implied by *bias* is overridden. Not that ``ddof=1`` will return the unbiased estimate and ``ddof=0`` will return the simple average.
    :return: The correlation matrix of the two timeseries with the time dimension as samples.
    :rtype: ``Tensor``

    Example:

        >>> x = echotorch.randn(5, time_length=100)
        >>> y = echotorch.randn(5, time_length=100)
        >>> echotorch.cor(x, y)
        tensor([[-0.1257,  0.3849, -0.2094, -0.2107,  0.1781],
                [-0.0990, -0.5916,  0.3169, -0.1333, -0.0315],
                [ 0.0443, -0.0571, -0.2228, -0.3075,  0.0995],
                [ 0.2477, -0.5867,  0.4337, -0.2673,  0.0725],
                [ 0.2607,  0.4544,  0.5199,  0.2562,  0.4110]])
    """
    # Get covariance matrix
    cov_m = cov(t1, t2, bias, ddof)

    # Get sigma for t1 and t2
    t1_std = torch.unsqueeze(tstd(t1), dim=1)
    t2_std = torch.unsqueeze(tstd(t2), dim=0)

    # Inner product of s(t1) and s(t2)
    t_inner = torch.mm(t1_std, t2_std)

    # Divide covariance matrix
    return torch.divide(cov_m, t_inner)
# end cor


# Covariance matrix
def cov(
        t1: TimeTensor,
        t2: Optional[TimeTensor] = None,
        bias: Optional[bool] = False,
        ddof: Optional[int] = None
) -> Tensor:
    r"""Returns the covariance matrix of two 1-D or 0-D timeseries with the same number of channels.

    As the size of the two timetensors is :math:`(T, p)`, the returned
    matrix :math:`C` has a size :math:`(p, p)`. Each element :math:`C_{ij}` of
    matrix :math:`C` is the covariance :math:`Cov(x_i, y_i)` between :math:`x_i` and :math:`y_i`,
    such that,

    .. math::
        :nowrap:

        $$
        C =
        \begin{pmatrix}
        \sigma_{x_{1}y_{1}} & \sigma_{x_{1}y_{2}} &  \cdots & \sigma_{x_{1}y_{p}} \\
        \sigma_{x_{2}y_{1}} & \ddots & \cdots & \vdots \\
        \vdots & \vdots & \ddots & \vdots \\
        \sigma_{x_{p}y_{1}} & \cdots & \cdots & \sigma_{x_{p}y_{p}}
        \end{pmatrix}
        $$

    where :math:`p` is the number of channels.

    :param t1: first timetensor containing the uni or multivariate timeseries. The time dimension should be at position 0.
    :type t1: ``TimeTensor``
    :param t2: An additional ``TimeTensor`` with same shape and time length. If ``None``, the auto-covariance matrix of *t1* is returned.
    :type t2: ``TimeTensor``, optional
    :param bias: Default normalization (False) is by :math:`(N - 1)`, where :math:`N` is the number of observations given (unbiased) or length of the timeseries. If *bias* is True, then normalization is by :math:`N`. These values can be overriden by using the keyword *ddof*.
    :type bias: ``bool``, optional
    :param ddof: If not *None* the default value implied by *bias* is overriden. Not that ``ddof=1`` will return the unbiased estimate and ``ddof=0`` will return the simple average.
    :return: The covariance matrix of the two timeseries with the time dimension as samples.
    :rtype: ``Tensor``

    Example:
        >>> x = echotorch.randn(5, time_length=100)
        >>> y = echotorch.randn(5, time_length=100)
        >>> echotorch.cov(x, y)
        tensor([[-0.0754, -0.0818, -0.0063, -0.0484,  0.0499],
                [ 0.0290,  0.2155,  0.0735,  0.2179, -0.0991],
                [ 0.0117,  0.0356, -0.0438,  0.0088, -0.0487],
                [ 0.0080,  0.0390, -0.0212,  0.0773,  0.1014],
                [-0.1000, -0.0774,  0.0011,  0.0819, -0.0735]])
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

    # Only 1-D or 0-D timetensors
    if t1.cdim > 1 or t2.cdim > 1 or t1.cdim != t2.cdim:
        raise ValueError(
            "Expected 1-D or 0-D timeseries, with same shape, but got {} and {}".format(t1.cdim, t2.cdim)
        )
    # end if

    # If 0-D, transform in 1-D
    if t1.cdim == 0:
        t1 = torch.unsqueeze(t1, dim=t1.time_dim+1)
        t2 = torch.unsqueeze(t2, dim=t2.time_dim + 1)
    # end if

    # Compute means
    t1_mean = tmean(t1)
    t2_mean = tmean(t2)

    # Bias value
    if ddof is None:
        add_bias = 1
        if bias:
            add_bias = 0
        # end if
    else:
        add_bias = ddof
    # end if

    # Compute covariance
    return mm((t1 - t1_mean).t(), t2 - t2_mean) / (t1.tlen - add_bias)
# end cov

