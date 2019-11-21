# -*- coding: utf-8 -*-
#
# File : papers/schaetti2016/transforms/Concat.py
# Description : Transform images to a concatenation of multiple transformations.
# Date : 11th of November, 2019
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
import torch
import torch.nn as nn
import echotorch.nn as etnn
import echotorch.nn.reservoir
import echotorch.nn.utils


# ESN with Join State
class ESNJS(etnn.Node):
    """
    ESN with Join State
    """

    # Constructor
    def __init__(self, image_size, hidden_dim, spectral_radius, leaky_rate, ridge_param, input_scaling,
                 debug=etnn.Node.NO_DEBUG, test_case=None):
        """
        Constructor
        :param image_size: Input image size
        :param hidden_dim: Reservoir size
        :param spectral_radius: Reservoir's spectral radius
        :param leaky_rate:
        :param ridge_param:
        :param debug:
        :param test_case:
        """
        # Super
        super(ESNJS, self).__init__(
            input_dim=image_size,
            output_dim=10,
            debug=debug,
            test_case=test_case,
            dtype=torch.float64
        )

        # Internal matrix
        w_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
            connectivity=0.1,
            spetral_radius=spectral_radius
        )

        # Input weights
        win_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
            connectivity=0.1,
            scale=input_scaling
        )

        # Bias vector
        wbias_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
            connectivity=0.1
        )

        # Create ESN
        self.esn = echotorch.nn.reservoir.LiESNCell(
            input_dim=image_size,
            output_dim=10,
            spectral_radius=spectral_radius,
            leaky_rate=leaky_rate,
            w=w_generator.generate(size=(hidden_dim, hidden_dim)),
            w_in=win_generator.generate(size=(hidden_dim, image_size)),
            w_bias=wbias_generator.generate(size=hidden_dim),
        )

        # Join states
        self.js = echotorch.nn.utils.JoinStates(
            input_dim=hidden_dim,
            join_length=image_size
        )

        # Ridge regression output
        self.output = echotorch.nn.linear.RRCell(
            input_dim=hidden_dim,
            output_dim=10,
            ridge_param=ridge_param,
            with_bias=True,
            softmax_output=True,
            averaged=True
        )

        # We train the RR layer
        self.finalized_element(self.output)
    # end __init__

    # Forward
    def forward(self, u):
        """
        Forward
        :param u:
        :return:
        """
        # Reservoir layer
        x = self.esn(u)

        # Join state
        x = self.js(x)

        # Output RR
        return self.output(x)
    # end forward

# end ESNJS
