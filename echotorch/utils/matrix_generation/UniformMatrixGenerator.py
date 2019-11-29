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
            input_set=[1.0, -1.0]
        )

        # Set parameters
        self._set_parameters(args=kwargs)
    # end __init__

    # Generate the matrix
    def generate(self, size, dtype=torch.float64):
        """
        Generate the matrix
        :param size: Matrix size
        :return: Generated matrix
        """
        # Params
        connectivity = self.get_parameter('connectivity')
        input_set = self.get_parameter('input_set')

        # Generate
        if connectivity is None:
            w = (np.random.randint(0, 2, size) * 2.0 - 1.0)
        else:
            sparsity = 1.0 - connectivity
            w = np.random.choice(
                np.append([0], input_set),
                size,
                p=np.append([1.0 - sparsity], [sparsity / len(input_set)] * len(input_set))
            )
        # end if

        # Transform to torch tensor
        if dtype == torch.float32:
            w = torch.from_numpy(w.astype(np.float32))
        else:
            w = torch.from_numpy(w.astype(np.float64))
        # end if

        # Scale
        w *= self.get_parameter('scale')

        # Set spectral radius
        if w.ndimension() == 2 and w.size(0) == w.size(1) and self.get_parameter('apply_spectral_radius'):
            w = (w / echotorch.utils.spectral_radius(w)) * self.get_parameter('spectral_radius')
        # end if

        return w
    # end generate

# end UniformMatrixGenerator


# Add
matrix_factory.register_generator("uniform", UniformMatrixGenerator)

