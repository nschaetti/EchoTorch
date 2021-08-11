# -*- coding: utf-8 -*-
#
# File : echotorch/matrices.py
# Description : EchoTorch matrix creation utility functions.
# Date : 30th of March, 2021
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
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>,
# University of Geneva <nils.schaetti@unige.ch>


# Imports
from typing import Tuple, Optional
import torch

# EchoTorch imports
import echotorch.utils.matrix_generation as etmg
from echotorch.utils.matrix_generation import MatrixGenerator


# Cycle matrix with jumps generator
def cycle_with_jumps_generator(
        *size: Tuple[int],
        cycle_weight: Optional[float] = 1.0,
        jump_weight: Optional[float] = 1.0,
        jump_size: Optional[float] = 2.0,
        dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    r"""Generate cycle matrix with jumps (Rodan and Tino, 2012)

    :param size: Size of the matrix
    :type size: Tuple of two ints
    :param cycle_weight:
    :type cycle_weight:
    :param jump_weight:
    :type jump_weight:
    :param jump_size:
    :type jump_size:
    :param dtype:
    :type dtype: ``torch.dtype``
    :return: Generated cycle matrix
    :rtype: ``torch.Tensor``

    """
    # The matrix must be a square matrix nxn
    if len(size) == 2 and size[0] == size[1]:
        # Matrix full of zeros
        w = torch.zeros(size, dtype=dtype)

        # How many neurons
        n_neurons = size[0]

        # Create the cycle
        w[0, -1] = cycle_weight
        for i in range(n_neurons):
            w[i, i - 1] = cycle_weight
        # end for

        # Create jumps
        for i in range(0, n_neurons - jump_size + 1, jump_size):
            w[i, (i + jump_size) % n_neurons] = jump_weight
            w[(i + jump_size) % n_neurons, i] = jump_weight
        # end for

        return w
    else:
        raise ValueError("The generated matrix must be a square matrix : {}".format(size))
    # end if
# end cycle_with_jumps_generator

