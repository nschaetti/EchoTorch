# -*- coding: utf-8 -*-
#
# File : echotorch/datasets/functional/random_processes.py
# Description : Examples of time series generation based on random processes
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
from typing import Any, List, Optional, Tuple, Union, Callable
import torch
import echotorch


# Random walk
def random_walk(
        size: int,
        length: int,
        shape: Optional[Union[List, Tuple]] = None,
        noise_mean: Optional[float] = 0.0,
        noise_std: Optional[float] = 1.0
) -> Tuple[echotorch.TimeTensor]:
    r"""Generate time series based on a random walk process.

    Definition
        (From Wikipedia) In mathematics, a **random walk** is a mathematical object, known as a stochastic or random
        process, that describes a path that consists of a succession of random steps on some mathematical space such
        as the integers.

        If :math:`x(t)` is the generated random walk at time *t* and :math:`z(t)` a white noise with mean
        :math:`\mu` (noise_mean) and a standard deviation :math:`\sigma` (noise_std), the :math:`x(t)` is described as

        .. math::
            x(t) = x({t-1}) + z(t)

        `Article on Wikipedia <https://en.wikipedia.org/wiki/Random_walk>`__

    :param size: how many samples to generate.
    :type size: ``int``
    :param length: length of generated time series.
    :type length: ``int``
    :param shape: shape of time series.
    :type shape: ``torch.Size``, ``list`` or ``tuple`` of ``int``
    :param noise_mean: mean :math:`\mu` of the white noise.
    :type noise_mean: ``float``
    :param noise_std: standard deviation :math:`\sigma` of the white noise.
    :type noise_std: ``float``
    :return: a list of :class:`TimeTensor` with series generated from random walk.
    :rtype: list of :class:`TimeTensor`

    Example:

        >>> echotorch.data.random_walk(1, length=10000, shape=(2, 2))
        timetensor(tensor([[[-2.1806e-01,  1.5221e-01],
                             [ 1.6733e-01, -9.5691e-01]],
                            [[ 6.9345e-01,  4.2999e-01],
                             [-1.8667e-01, -2.5323e-01]],
                            [[ 4.9236e-01, -2.5215e+00],
                             [-1.5146e-01,  1.5272e+00]],
                            ...,
                            [[ 1.6925e+02, -9.9522e+01],
                             [ 9.7266e-01,  2.2402e+01]],
                            [[ 1.7010e+02, -1.0009e+02],
                             [ 1.0102e+00,  2.3406e+01]],
                            [[ 1.7160e+02, -1.0048e+02],
                             [ 7.4558e-01,  2.4151e+01]]]), time_dim: 0)
    """
    # Samples
    samples = list()

    # Shape
    shape = () if shape is None else shape

    # For each sample
    for n in range(size):
        # Generate noise Zt
        zt_noise = echotorch.randn(*shape, time_length=length+1) * noise_std + noise_mean

        # Space for xt
        xt = echotorch.zeros(*shape, time_length=length)

        # x(0)
        xt[0] = zt_noise[0]

        # For each timestep
        for t in range(1, length):
            xt[t] = xt[t-1] + zt_noise[t]
        # end for

        # Add
        samples.append(xt)
    # end for

    return samples
# end random_walk


# Random walk
def rw(
        size: int,
        length: int,
        shape: Union[torch.Size, List, Tuple],
        noise_mean: Optional[float] = 0.0,
        noise_std: Optional[float] = 1.0
) -> Tuple[echotorch.TimeTensor]:
    r"""Alias for :func:`echotorch.data.random_walk`.
    """
    return random_walk(
        size=size,
        length=length,
        shape=shape,
        noise_mean=noise_mean,
        noise_std=noise_std
    )
# end rw


# Univariate Random Walk
def unirw(
        size: int,
        length: int,
        noise_mean: Optional[float] = 0.0,
        noise_std: Optional[float] = 1.0
) -> Tuple[echotorch.TimeTensor]:
    r"""Generate a univariate time series based on a random walk process.

    See :func:`echotorch.data.random_walk` for mathematical details.

    :param size: how many samples to generate.
    :type size: ``int``
    :param length: lenght of generated time series.
    :type length: ``int``
    :param noise_mean: mean :math:`\mu` of the white noise.
    :type noise_mean: ``float``
    :param noise_std: standard deviation :math:`\sigma` of the white noise.
    :type noise_std: ``float``
    :return: a list of :class:`TimeTensor` with series generated from random walk.
    :rtype: list of :class:`TimeTensor`

    Example:

        >>> echotorch.data.unirw(1, length=10000)
        timetensor(tensor([ -1.5256,  -2.2758,  -2.9298,  ..., -37.9416, -36.9469, -38.1765]), time_dim: 0)
    """
    return random_walk(
        size=size,
        length=length,
        shape=(),
        noise_mean=noise_mean,
        noise_std=noise_std
    )
# end unirw


# Multivariate Moving average
def moving_average(
        samples: int,
        length: int,
        order: Optional[int] = None,
        size: Optional[int] = None,
        theta: Optional[torch.Tensor] = None,
        noise_mean: Optional[float] = 0.0,
        noise_std: Optional[float] = 1.0,
        noise_func: Optional[Callable] = echotorch.randn,
        parameters_func: Optional[Callable] = torch.rand
) -> List[echotorch.TimeTensor]:
    r"""Create multivariate time series based on the moving average model (MA) or
    vector moving average process (VMA).

    The multivariate form of the Moving Average model MA(q) of order :math:`q` is of
    the form

    .. math::
        x(t) = z(t) + \Theta_1 z(t-1) + \dots + \Theta_q z(t-q)

    :math:`q` is the number of last entries used for the average. :math:`x(t)` is the moving average output at time
    *t* and :math:`z(t)` a noise with mean :math:`\mu` (*noise_mean*) and standard deviation :math:`\sigma` (*noise_std*).
    This function implements the simple moving average where past entries are equally weighted.

    For Weighed Moving Average (WMA) see :func:`echotorch.data.weighted_moving_average`.

    For Cumulative Moving Average (CMA) see :func:`echotorch.data.cumulative_moving_average`.

    For Exponential Moving Average (EMA) see :func:`echotorch.data.exponential_moving_average`.

    `Article on Wikipedia <https://en.wikipedia.org/wiki/Moving_average>`__

    :param samples: how many samples to generate.
    :type samples: ``ìnt``
    :param length: length of the time series to generate.
    :type length: ``ìnt``
    :param order: value of of :math:`q`, the order of the moving average :math:`MA(q)`.
    :type order: ``ìnt``
    :param size: number of variables in the output time series.
    :type size: ``ìnt``
    :param theta: a tensor of size (order, size, size) containing parameter for each timestep as a matrix.
    :type theta: ``torch.Tensor``
    :param noise_mean: mean :math:`\mu` of the white noise
    :type noise_mean: ``float``
    :param noise_std: standard deviation :math:`\Sigma` of the white noise
    :type noise_std: ``float``
    :param noise_func: callable object to generate noise compatible with echotorch creation operator interace.
    :type noise_func: ``callable``

    Example:

        >>> moving_average = echotorch.data.moving_average(1, length=200, order=30, size=1)
        >>> plt.figure()
        >>> echotorch.viz.timeplot(moving_average[0], title="Multivariate Moving Average MA(q)")
        >>> plt.show()

    """
    # Check that parameters or theta or given
    if (order is None or size is None) and theta is None:
        raise ValueError(
            "Order and size, or theta must at least be given (here {}, {} and {}".format(order, size, theta)
        )
    # end if

    # Check theta size if given
    if theta is not None:
        # 3D tensor
        if theta.ndim != 3:
            raise ValueError(
                "Expected 3D tensor for theta with size (order, size, size), but {}D given".format(theta.ndim)
            )
        # end if

        # First two dim are square
        if theta.size()[1] != theta.size()[2]:
            raise ValueError(
                "Expected 3D tensor with first two dimension squared (order, size, size), "
                "but tensor of shape {} given".format(theta.size())
            )
        # end if
    # end if

    # Order, number of variables
    s = samples
    q = theta.size()[0] if theta is not None else order
    n = theta.size()[1] if theta is not None else size

    # If theta null, generate parameters
    if theta is None: theta = parameters_func(q, n, n)

    # Add identity for t
    theta = torch.cat((torch.unsqueeze(torch.eye(n), 0), theta), dim=0)

    # Samples
    samples = list()

    # For each sample
    for s_i in range(s):
        # Generate noise Zt
        zt = noise_func(n, length=length + q) * noise_std + noise_mean

        # Space for output
        xt = echotorch.zeros(n, length=length)

        # For each timestep
        for t in range(length):
            xt[t] = sum([torch.mv(theta[k], zt[t+q-k]) for k in range(0, q+1)])
        # end for

        # Add
        samples.append(xt)
    # end for

    return samples
# end moving_average


# Multivariate Moving average
def ma(
        samples: int,
        length: int,
        order: Optional[int] = None,
        size: Optional[int] = None,
        theta: Optional[torch.Tensor] = None,
        noise_mean: Optional[float] = 0.0,
        noise_std: Optional[float] = 1.0,
        noise_func: Optional[Callable] = echotorch.randn,
        parameters_func: Optional[Callable] = torch.rand
) -> List[echotorch.TimeTensor]:
    r"""Alias for :func:`echotorch.data.moving_average`.
    """
    return moving_average(
        samples=samples,
        length=length,
        order=order,
        size=size,
        theta=theta,
        noise_mean=noise_mean,
        noise_std=noise_std,
        noise_func=noise_func,
        parameters_func=parameters_func
    )
# end ma


# Univariate Moving Average
def unima(
        samples: int,
        length: int,
        order: Optional[int] = None,
        noise_mean: Optional[float] = 0.0,
        noise_std: Optional[float] = 1.0,
        noise_func: Optional[Callable] = echotorch.randn,
        parameters_func: Optional[Callable] = torch.rand
) -> List[echotorch.TimeTensor]:
    r"""Returns a univariate time series based on the moving average model (MA).

    Key arguments:
        * **theta** (``list of float``) - parameters for each timestep as a float.

    See :func:`echotorch.data.moving_average` for more details.

    Example:

        >>> ...
    """
    # Generate series
    ma_series = moving_average(
        samples=samples,
        length=length,
        order=order,
        size=1,
        noise_mean=noise_mean,
        noise_std=noise_std,
        noise_func=noise_func,
        parameters_func=parameters_func
    )

    # To 0-D
    for i in range(len(ma_series)):
        ma_series[i] = ma_series[i][:, 0]
    # end for

    return ma_series
# end unima


# Multivariate Weighed Moving Average (WMA)
def weighted_moving_average() -> List[echotorch.TimeTensor]:
    r"""Create multivariate time series based on the weighted moving average model (WMA).
    """
    pass
# end weighted_moving_average


# Alias for weighted_moving_average
def wma() -> List[echotorch.TimeTensor]:
    r"""Alias for :func:`echotorch.data.weighted_moving_average`.
    """
    pass
# end wma


# Multivariate Cumulative Average (CMA)
def cumulative_moving_average() -> List[echotorch.TimeTensor]:
    r"""Create multivariate time series based on the cumulative moving average model (CMA).
    """
    pass
# end cumulative_moving_average


# Alias for cumulative_moving_average
def cma() -> List[echotorch.TimeTensor]:
    r"""Alias for :func:`echotorch.data.cumulative_moving_average`.
    """
    pass
# end cma


# Exponential Moving Average (EMA)
def exponential_moving_average() -> List[echotorch.TimeTensor]:
    r"""Create multivariate time series based on the exponential moving average model (EMA).
    """
    pass
# end exponential_moving_average


# Alias for exponential_moving_average
def ema() -> List[echotorch.TimeTensor]:
    r"""Alias for :func:`echotorch.data.exponential_moving_average`.
    """
    pass
# end ema


# Multivariate Autoregressive process
def ar(
        samples: int,
        length: int,
        order: Optional[int] = None,
        size: Optional[int] = None,
        phi: Optional[torch.Tensor] = None,
        noise_mean: Optional[float] = 0.0,
        noise_std: Optional[float] = 1.0,
        noise_func: Optional[Callable] = echotorch.randn,
        parameters_func: Optional[Callable] = torch.rand
) -> List[echotorch.TimeTensor]:
    r"""Alias for :func:`echotorch.data.autoregressive_process`.
    """
    return autoregressive_process(
        samples=samples,
        length=length,
        order=order,
        size=size,
        phi=phi,
        noise_mean=noise_mean,
        noise_std=noise_std,
        noise_func=noise_func,
        parameters_func=parameters_func
    )
# end ar


# Multivariate Auto-regressive process
def autoregressive_process(
        samples: int,
        length: int,
        order: Optional[int] = None,
        size: Optional[int] = None,
        phi: Optional[torch.Tensor] = None,
        noise_mean: Optional[float] = 0.0,
        noise_std: Optional[float] = 1.0,
        noise_func: Optional[Callable] = echotorch.randn,
        parameters_func: Optional[Callable] = torch.rand
) -> List[echotorch.TimeTensor]:
    r"""Create uni or multivariate time series based on autoregressive process (AR) or
    vector autoregressive model (AR).
    """
    # Check that parameters or theta or given
    if (order is None or size is None) and phi is None:
        raise ValueError(
            "Order and size, or theta must at least be given (here {}, {} and {}".format(order, size, theta)
        )
    # end if

    # Check theta size if given
    if phi is not None:
        # 3D tensor
        if phi.ndim != 3:
            raise ValueError(
                "Expected 3D tensor for theta with size (order, size, size), but {}D given".format(phi.ndim)
            )
        # end if

        # First two dim are square
        if phi.size()[1] != phi.size()[2]:
            raise ValueError(
                "Expected 3D tensor with first two dimension squared (order, size, size), "
                "but tensor of shape {} given".format(phi.size())
            )
        # end if
    # end if

    # Order, number of variables
    s = samples
    p = phi.size()[0] if phi is not None else order
    n = phi.size()[1] if phi is not None else size

    # If theta null, generate parameters
    if phi is None:
        phi = parameters_func(p, n, n)
        phi /= torch.sum(phi, dim=0)
    # end if

    # Add identity for t
    # phi = torch.cat((torch.unsqueeze(torch.eye(n), 0), phi), dim=0)

    # Samples
    samples = list()

    # For each sample
    for s_i in range(s):
        # Generate noise Zt
        zt = noise_func(n, time_length=length) * noise_std + noise_mean

        # Space for output
        xt = echotorch.zeros(n, time_length=length)

        # For each timestep
        for t in range(length):
            xt[t] = zt[t]
            xt[t] += sum([torch.mv(phi[k], xt[t - k]) for k in range(0, p) if t - k >= 0])
        # end for

        # Add
        samples.append(xt)
    # end for

    return samples
# end autoregressive_process


# Alias to autoregressive_moving_average
def arma(
        samples: int,
        length: int,
        regressive_order: Optional[int] = None,
        moving_average_order: Optional[int] = None,
        size: Optional[int] = None,
        theta: Optional[torch.Tensor] = None,
        phi: Optional[torch.Tensor] = None,
        noise_mean: Optional[float] = 0.0,
        noise_std: Optional[float] = 1.0,
        noise_func: Optional[Callable] = echotorch.randn,
        parameters_func: Optional[Callable] = torch.rand
) -> List[echotorch.TimeTensor]:
    r"""Alias to :func:`echotorch.data.autoregressive_moving_average`.
    """
    return autoregressive_moving_average(
        samples=samples,
        length=length,
        regressive_order=regressive_order,
        moving_average_order=moving_average_order,
        size=size,
        theta=theta,
        phi=phi,
        noise_mean=noise_mean,
        noise_std=noise_std,
        noise_func=noise_func,
        parameters_func=parameters_func
    )
# end arma


# Multivariate AutoRegressive Moving Average process (ARMA)
def autoregressive_moving_average(
        samples: int,
        length: int,
        regressive_order: Optional[int] = None,
        moving_average_order: Optional[int] = None,
        size: Optional[int] = None,
        theta: Optional[torch.Tensor] = None,
        phi: Optional[torch.Tensor] = None,
        noise_mean: Optional[float] = 0.0,
        noise_std: Optional[float] = 1.0,
        noise_func: Optional[Callable] = echotorch.randn,
        parameters_func: Optional[Callable] = torch.rand
) -> List[echotorch.TimeTensor]:
    r"""Create uni or multivariate time series based on AutoRegressive Moving Average process (ARMA) or
    Vector ARMA  (ARMAV).

    :param samples: How many samples to generate.
    :type samples: ``ìnt``
    :param length: Length of the time series to generate.
    :type length: ``ìnt``
    :param order: Value of of :math:`q`, the order of the moving average :math:`MA(q)`.
    :type order: ``ìnt``
    :param size: Number of variables in the output time series.
    :type size: ``ìnt``
    :param theta: A tensor of size (order, size, size) containing parameter for each timestep as a matrix.
    :type theta: ``torch.Tensor``
    :param noise_mean: Mean :math:`\mu` of the white noise
    :type noise_mean: ``float``
    :param noise_std: Standard deviation :math:`\Sigma` of the white noise
    :type noise_std: ``float``
    :param noise_func: Callable object to generate noise compatible with echotorch creation operator interace.
    :type noise_func: ``callable``

    """
    # Check that parameters or theta or given
    if (regressive_order is None or moving_average_order is None or size is None) \
            and phi is None and theta is None:
        raise ValueError(
            "Regressive order, moving average order and size, or theta must at least be "
            "given (here {}, {} and {}".format(regressive_order, moving_average_order, size, theta)
        )
    # end if

    # Check theta size if given
    if phi is not None and theta is not None:
        # 3D tensor
        if phi.ndim != 3 or theta.ndim != 3:
            raise ValueError(
                "Expected 3D tensor for theta and phi with size (order, size, size), but {}D given".format(phi.ndim)
            )
        # end if

        # First two dim are square
        if phi.size()[1] != phi.size()[2] or theta.size()[1] != theta.size()[2] or theta.size()[1] != phi.size()[1]:
            raise ValueError(
                "Expected 3D tensor with first two dimension squared (order, size, size) and equal, "
                "but tensors of shape {} and {} given".format(phi.size(), theta.size())
            )
        # end if
    # end if

    # Order, number of variables
    s = samples
    p = phi.size()[0] if phi is not None else regressive_order
    q = theta.size()[0] if theta is not None else moving_average_order
    n = phi.size()[1] if phi is not None else size

    # If phi null, generate parameters
    if phi is None:
        phi = parameters_func(p, n, n)
        phi /= torch.sum(phi, dim=0)
    # end if

    # If phi null, generate parameters
    if theta is None: theta = parameters_func(q, n, n)

    # Add identity for t
    # theta = torch.cat((torch.unsqueeze(torch.eye(n), 0), theta), dim=0)

    # Samples
    samples = list()

    # For each sample
    for s_i in range(s):
        # Generate noise Zt
        # zt = noise_func(n, time_length=length + q) * noise_std + noise_mean
        zt = noise_func(n, time_length=length + q) * noise_std + noise_mean

        # Space for output
        xt = echotorch.zeros(n, time_length=length)

        # For each timestep
        for t in range(length):
            xt[t] = zt[t]
            xt[t] += sum([torch.mv(phi[k], xt[t - k]) for k in range(0, p) if t - k >= 0])
            xt[t] += sum([torch.mv(theta[k-1], zt[t+q-k]) for k in range(1, q+1)])
        # end for

        # Add
        samples.append(xt)
    # end for

    return samples
# end autoregressive_moving_average
