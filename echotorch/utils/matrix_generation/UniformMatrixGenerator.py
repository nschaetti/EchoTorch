# -*- coding: utf-8 -*-
#
# File : echotorch/utils/matrix_generation/UniformMatrixGenerator.py
# Description : Generate matrix it uniformally distributed weights.
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
import numpy as np
import echotorch.utils
from .MatrixGenerator import MatrixGenerator
from .MatrixFactory import matrix_factory
import warnings


# Generate matrix it uniformly distributed weights.
class UniformMatrixGenerator(MatrixGenerator):
    """
    Generate matrix it uniformly distributed weights.
    """

    # Constructor
    def __init__(self, **kwargs):
        """
        Constructor
        :param kwargs: Parameters of the generator
        """
        # Set default parameter values
        super(UniformMatrixGenerator, self).__init__(
            connectivity=1.0,
            spectral_radius=0.99,
            apply_spectral_radius=True,
            scale=1.0,
            input_set=[1.0, -1.0],
            minimum_edges=0,
            min=-1.0,
            max=1.0
        )

        # Set parameters
        self._set_parameters(args=kwargs)
    # end __init__

    #region PRIVATE

    # Generate the matrix
    def _generate_matrix(self, size, dtype=torch.float64):
        """
        Generate the matrix
        :param size: Matrix size
        :return: Generated matrix
        """
        # Params
        connectivity = self.get_parameter('connectivity')
        input_set = self.get_parameter('input_set')

        # If not connectivity, then its 1.0
        if connectivity is None:
            connectivity = 1.0
        # end if

        # Generate
        if input_set is None:
            # Generate matrix with entries from norm
            w = torch.zeros(size, dtype=dtype)
            w = w.uniform_(self.get_parameter('min'), self.get_parameter('max'))
        else:
            # Generate from choice
            w = np.random.choice(
                input_set,
                size,
                p=[1.0 / len(input_set)] * len(input_set)
            )

            # Transform to torch tensor
            if dtype == torch.float32:
                w = torch.from_numpy(w.astype(np.float32))
            else:
                w = torch.from_numpy(w.astype(np.float64))
            # end if
        # end if

        # Generate mask from bernoulli
        mask = torch.bernoulli(torch.zeros(size, dtype=dtype).fill_(connectivity))

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

        return w
    # end _generate_matrix

    #endregion PRIVATE

# end UniformMatrixGenerator


# Add
matrix_factory.register_generator("uniform", UniformMatrixGenerator)

