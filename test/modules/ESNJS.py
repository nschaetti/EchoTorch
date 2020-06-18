# -*- coding: utf-8 -*-
#
# File : test/modules/ESNJS.py
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
    def __init__(self, image_size, input_dim, hidden_dim, leaky_rate, ridge_param, w_generator, win_generator,
                 wbias_generator, input_scaling=1.0, debug=etnn.Node.NO_DEBUG, test_case=None, dtype=torch.float64):
        """
        Constructor
        :param image_size:
        :param input_dim:
        :param hidden_dim:
        :param leaky_rate:
        :param ridge_param:
        :param w_generator:
        :param win_generator:
        :param wbias_generator:
        :param input_scaling:
        :param debug:
        :param test_case:
        :param dtype:
        """
        # Super
        super(ESNJS, self).__init__(
            input_dim=input_dim,
            output_dim=10,
            debug=debug,
            test_case=test_case,
            dtype=dtype
        )

        # Create ESN
        self.esn = echotorch.nn.reservoir.LiESNCell(
            input_dim=input_dim,
            output_dim=hidden_dim,
            leaky_rate=leaky_rate,
            input_scaling=input_scaling,
            w=w_generator.generate(size=(hidden_dim, hidden_dim)),
            w_in=win_generator.generate(size=(hidden_dim, input_dim)),
            w_bias=wbias_generator.generate(size=hidden_dim),
            dtype=dtype
        )

        # Join states
        self.js = echotorch.nn.utils.JoinStates(
            input_dim=hidden_dim,
            join_length=image_size,
            dtype=dtype
        )

        # Ridge regression output
        self.output = echotorch.nn.linear.RRCell(
            input_dim=hidden_dim * image_size,
            output_dim=10,
            ridge_param=ridge_param,
            with_bias=True,
            softmax_output=True,
            averaged=True,
            dtype=dtype
        )

        # We train the RR layer
        self.add_trainable(self.output)
    # end __init__

    # Forward
    def forward(self, *input, **kwargs):
        """
        Forward
        :param u:
        :return:
        """
        # Inputs and targets
        u, yh = input

        # Reservoir layer
        x = self.esn(u)

        # Join state
        x = self.js(x)

        # Output RR
        return self.output(x, yh)
    # end forward

# end ESNJS
