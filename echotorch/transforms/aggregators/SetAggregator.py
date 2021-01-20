# -*- coding: utf-8 -*-
#
# File : echotorch/transforms/aggregators/SetAggregator.py
# Description : Join multiple aggregators in one.
# Date : 19th of January, 2021
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


# A set of aggregators
class SetAggregator(Aggregator):
    """
    An aggregator which join multiple aggregators in a set.
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, aggregators, **kwargs):
        """
        Constructor
        :param n_lags:
        :param kwargs:
        """
        # To Aggregator
        super(SetAggregator, self).__init__(**kwargs)

        # Properties
        self._aggregators = aggregators
    # end __init__

    # endregion CONSTRUCTOR

    # region PRIVATE

    # endregion PRIVATE

    # region OVERRIDE

    # Initialize
    def _initialize(self):
        """
        Initialize
        """
        self._initialized = True
    # end _initialize

    # Aggregate information
    def _aggregate(self, x):
        """
        Aggregate information
        :param x: Input tensor data
        """
        # Pass data to each aggregators
        for agg in self._aggregators:
            agg(x)
        # end for
    # end _aggregate

    # Finalize
    def _finalize(self):
        """
        Finalize aggregation
        """
        # Finalize each aggregator
        for agg in self._aggregators:
            agg.finalize()
        # end for

        # For each
        self._finalized = True
    # endregion OVERRIDE

# end SetAggregator
