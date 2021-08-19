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
        shape: Union[torch.Size, List, Tuple],
        noise_mean: Optional[float] = 0.0,
        noise_std: Optional[float] = 1.0
) -> Tuple[echotorch.TimeTensor]:
    r"""Generate time series based on a random walk process.

    Definition
        (From Wikipedia) In mathematics, a **random walk** is a mathematical object, known as a stochastic or random
        process, that describes a path that consists of a succession of random steps on some mathematical space such
        as the integers.

        If :math:`X_t` is the generated random walk at time *t* and :math:`Z_t` a white noise with mean
        :math:`\mu` (noise_mean) and a standard deviation `\Sigma` (noise_std), the :math:`X_t` is described as

        .. math::
            x(t) = x({t-1}) + z(t)

        `Article on Wikipedia <https://en.wikipedia.org/wiki/Random_walk>`__

    :param size: How many samples to generate
    :type size: ``int``
    :param length: Length of generated time series
    :type length: ``int``
    :param shape: Shape of time series
    :type shape: ``torch.Size``, ``list`` or ``tuple`` of ``int``
    :param noise_mean: Mean :math:`\mu` of the white noise
    :type noise_mean: ``float``
    :param noise_std: Standard deviation :math:`\Sigma` of the white noise
    :type noise_std: ``float``
    :return: A list of ``TimeTensor`` with series generated from random walk
    :rtype: Tuple of ``TimeTensor``

    """
    # Samples
    samples = list()

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
    r"""Create uni or multivariate time series based on the moving average model (MA) or
    vector moving average process (VMA).

    The multivariate form of the Moving Average model MA(q) of order :math:`q` is of
    the form

    .. math::
        x(t) = z(t) + \Theta_1 z(t-1) + \dots + \Theta_q z(t-q)

    with :math:`\mathbf{q}(t)`

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
        zt = noise_func(n, time_length=length + q) * noise_std + noise_mean

        # Space for output
        xt = echotorch.zeros(n, time_length=length)

        # For each timestep
        for t in range(length):
            xt[t] = sum([torch.mv(theta[k], zt[t+q-k]) for k in range(0, q+1)])
        # end for

        # Add
        samples.append(xt)
    # end for

    return samples
# end moving_average


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
    if phi is None: phi = parameters_func(p, n, n)

    # Add identity for t
    phi = torch.cat((torch.unsqueeze(torch.eye(n), 0), phi), dim=0)

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

