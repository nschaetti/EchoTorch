# -*- coding: utf-8 -*-
#
# File : echotorch/utils/matrix_generation/CycleWithJumpsMatrixGenerator.py
# Description : Generate matrix as defined in Rodan and Tino, 2012.
# Date : 16th of July, 2020
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

# Import
import torch
import echotorch.utils
import numpy as np
from .MatrixGenerator import MatrixGenerator
from .MatrixFactory import matrix_factory


# Generate cycle matrix with jumps (Rodan and Tino, 2012)
class CycleWithJumpsMatrixGenerator(MatrixGenerator):
    """
    Generate cycle matrix with jumps (Rodan and Tino, 2012)
    """

    # Constructor
    def __init__(self, **kwargs):
        """
        Constructor
        :param kwargs: Parameters of the generator
        """
        # Set default parameter values
        super(CycleWithJumpsMatrixGenerator, self).__init__(
            connectivity=1.0,
            spectral_radius=0.99,
            apply_spectral_radius=False,
            scale=1.0,
            cycle_weight=1.0,
            jump_weight=1.0,
            jump_size=2.0
        )

        # Set parameters
        self._set_parameters(args=kwargs)
    # end __init__

    #region PRIVATE

    # Generate the matrix
    def _generate_matrix(self, size, dtype=torch.float64):
        """
        Generate the matrix
        :param: Matrix size (row, column)
        :param: Data type to generate
        :return: Generated matrix
        """
        # The matrix must be a square matrix nxn
        if size[0] == size[1]:
            # Params
            cycle_weight = self.get_parameter('cycle_weight')
            jump_weight = self.get_parameter('jump_weight')
            jump_size = self.get_parameter('jump_size')

            # Matrix full of zeros
            w = torch.zeros(size, dtype=dtype)

            # How many neurons
            n_neurons = size[0]

            # Create the cycle
            w[0, -1] = cycle_weight
            for i in range(n_neurons):
                w[i, i-1] = cycle_weight
            # end for

            # Create jumps
            for i in range(0, n_neurons - jump_size + 1, jump_size):
                w[i, (i + jump_size) % n_neurons] = jump_weight
                w[(i + jump_size) % n_neurons, i] = jump_weight
            # end for

            return w
        else:
            raise Exception("The generated matrix must be a square matrix : {}".format(size))
        # end if
    # end _generate_matrix

    #endregion PRIVATE

# end CycleWithJumpsMatrixGenerator


# Add
matrix_factory.register_generator("cycle_with_jumps", CycleWithJumpsMatrixGenerator)

