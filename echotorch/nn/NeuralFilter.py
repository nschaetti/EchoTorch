# -*- coding: utf-8 -*-
#
# File : echotorch/nn/NeuralFilter.py
# Description : Neural filters are special nodes which filter neural activities from layers.
# Date : 25th of November, 2019
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

"""
Created on 25 November 2019
@author: Nils Schaetti
"""

import torch
import torch.sparse
import torch.nn as nn
import numpy as np
from .Node import Node


# Base node for neural filters.
# Neural filters are special node to
# filter neural output activities from
# layers.
class NeuralFilter(Node):
    """
    Base node for neural filters.
    Neural filters are special node to
    filter neural output activities from
    layers.
    """

    # Constructor
    def __init__(self, input_dim, output_dim=0, *args, **kwargs):
        """
        Constructor
        :param input_dim: Filter's dimension.
        """
        # Superclass
        super(NeuralFilter, self).__init__(
            input_dim=input_dim,
            output_dim=input_dim,
            *args,
            **kwargs
        )
    # end __init__

    #######################
    # Properties
    #######################

    #######################
    # Forward/Backward/Init
    #######################

    # Filter
    def filter_transform(self, X, *args, **kwargs):
        """
        Filter signal
        :param X: Signal to filter
        :param kwargs: Options
        :return: Filtered signal
        """
        return X
    # end filter_transform

    # Train the filter
    def filter_fit(self, X, *args, **kwards):
        """
        Train filter
        :param X: Signal to learn from
        :param kwards: Options
        :return: Original signal
        """
        return X
    # end filter_fit

    # Forward
    def forward(self, *x, **kwargs):
        """
        Forward
        :param x: Inputs to filter
        :param kwargs: Options
        :return: Filtered neural activity
        """
        if self.training:
            return self.filter_fit(*x, **kwargs)
        else:
            return self.filter_transform(*x, **kwargs)
        # end if
    # end forward

    #######################
    # Public
    #######################

    #######################
    # Private
    #######################

    ###################
    # OVERLOAD
    ###################

# end Node
