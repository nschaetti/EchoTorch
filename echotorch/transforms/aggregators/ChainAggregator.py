# -*- coding: utf-8 -*-
#
# File : echotorch/transforms/aggregators/ChainAggregator.py
# Description : Run aggregators and transformers in chain
# Date : 20th of January, 2021
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

# EchoTorch imports
from echotorch.transforms import Aggregator


# Chain Aggregator
class ChainAggregator(Aggregator):
    """
    Run aggregators and transformers in chain
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, input_dim, children, *args, **kwargs):
        """
        Constructor
        """
        # Super constructor
        super(ChainAggregator, self).__init__(input_dim, *args, **kwargs)

        # Properties
        self._children = children
    # end __init__

    # endregion CONSTRUCTORS

    # region PROPERTIES

    # Children
    @property
    def children(self):
        """
        Children
        """
        return self._children
    # end children

    # endregion PROPERTIES

    # region OVERRIDE

    # Initialize, do nothing
    def _initialize(self):
        """
        Initialize
        """
        pass
    # end _initialize

    # Aggregate information
    def _aggregate(self, x):
        """
        Aggregate information
        """
        # Copy x
        xc = x.detach().clone()

        # Run each agg/trans
        for child in self._children:
            xc = child(xc)
        # end for

        return x
    # end _aggregate

    # endregion OVERRIDE

# end ChainAggregator
