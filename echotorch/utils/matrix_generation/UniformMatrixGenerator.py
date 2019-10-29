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
from .MatrixGenerator import MatrixGenerator
from .MatrixFactory import matrix_factory


# Generate matrix it uniformly distributed weights.
class UniformMatrixGenerator(MatrixGenerator):
    """
    Generate matrix it uniformly distributed weights.
    """

    # Generate the matrix
    def generate(self, size):
        """
        Generate the matrix
        :param size: Matrix size
        :return: Generated matrix
        """
        # Params
        try:
            connectivity = self._parameters['connectivity']
            size = self._parameters['size']
            dtype = self._parameters['dtype']
            input_set = self._parameters['input_set']
        except KeyError:
            raise Exception("Argument missing")
        # end try

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

        return w
    # end generate

# end UniformMatrixGenerator


# Add
matrix_factory.register_generator("uniform", UniformMatrixGenerator)

