# -*- coding: utf-8 -*-
#
# File : echotorch/utils/matrix_generation/NormalMatrixGenerator.py
# Description : Generate matrix it normally distributed weights.
# Date : 29th of October, 2019
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
import warnings


# Generate matrix it normally distributed weights.
class NormalMatrixGenerator(MatrixGenerator):
    """
    Generate matrix it normally distributed weights.
    """

    # Constructor
    def __init__(self, **kwargs):
        """
        Constructor
        :param kwargs: Parameters of the generator
        """
        # Set default parameter values
        super(NormalMatrixGenerator, self).__init__(
            connectivity=1.0,
            spectral_radius=0.99,
            apply_spectral_radius=True,
            scale=1.0,
            mean=0.0,
            std=1.0,
            minimum_edges=0
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
        # Params
        connectivity = self.get_parameter('connectivity')
        mean = self.get_parameter('mean')
        std = self.get_parameter('std')

        # Full connectivity if none
        if connectivity is None:
            w = torch.zeros(size, dtype=dtype)
            w = w.normal_(mean=mean, std=std)
        else:
            # Generate matrix with entries from norm
            w = torch.zeros(size, dtype=dtype)
            w = w.normal_(mean=mean, std=std)

            # Generate mask from bernoulli
            mask = torch.zeros(size, dtype=dtype)
            mask.bernoulli_(p=connectivity)

            # Minimum edges
            minimum_edges = min(self.get_parameter('minimum_edges'), np.prod(size))

            # Add edges until minimum is ok
            while torch.sum(mask) < minimum_edges:
                # Random position at 1
                x = torch.randint(high=size[0], size=(1, 1))[0, 0].item()
                y = torch.randint(high=size[1], size=(1, 1))[0, 0].item()
                mask[x, y] = 1.0
            # end while

            # Mask filtering
            w *= mask
        # end if

        return w
    # end _generate_matrix

    #enregion PRIVATE

# end NormalMatrixGenerator


# Add
matrix_factory.register_generator("normal", NormalMatrixGenerator)

