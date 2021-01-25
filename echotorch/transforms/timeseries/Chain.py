# -*- coding: utf-8 -*-
#
# File : echotorch/transforms/timeseries/Chain.py
# Description : Run transformers in chain
# Date : 25th of January, 2021
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
# Copyright Nils Schaetti <nils.schaetti@unine.ch>, <nils.schaetti@unige.ch>


# Imports
import torch

# EchoTorch imports
from echotorch.transforms import Transformer


# Chain
class Chain(Transformer):
    """
    Run transformers in chain
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, children, *args, **kwargs):
        """
        Constructor
        """
        # Super constructor
        super(Chain, self).__init__(
            *args,
            input_dim=children[0].input_dim,
            output_dim=children[-1].output_dim,
            **kwargs
        )

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

    # Transform data
    def _transform(self, x):
        """
        Transform information
        """
        # Run each agg/trans
        for child in self._children:
            x = child(x)
        # end for

        return x
    # end _transform

    # endregion OVERRIDE

# end Chain
