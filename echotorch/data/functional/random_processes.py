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
from typing import Any, List, Optional, Tuple, Union
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
            X_t = X_{t-1} + Z_t

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
        zt_noise = echotorch.randn(shape, length+1) * noise_std + noise_mean

        # Room for xt
        xt = echotorch.zeros(shape, length)

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

