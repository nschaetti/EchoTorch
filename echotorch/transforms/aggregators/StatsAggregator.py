# -*- coding: utf-8 -*-
#
# File : echotorch/transforms/ToOneHot.py
# Description : Transform integer targets to one-hot vectors.
# Date : 21th of November, 2019
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


# Basic statistics
class StatsAggregator(Aggregator):
    """
    An aggregator which compute the basic statistics about time series
    """

    # region PUBLIC

    # Get statistics
    def get_statistics(self, stat_type):
        """
        Get statistics
        """
        return self._data[stat_type] / self._counters[stat_type]
    # end get_statistics

    # endregion PUBLIC

    # region PRIVATE

    # Update entry
    def _update_entry(self, entry_name, value):
        """
        Update entry
        """
        self._data[entry_name] += value
        self._inc(entry_name)
    # end _update_entry

    # endregion PRIVATE

    # region OVERRIDE

    # Initialize
    def _initialize(self):
        """
        Initialize aggregators
        """
        self._register("mean", torch.zeros(self._input_dim))
        self._register("std", torch.zeros(self._input_dim))
        self._register("mean_length", 0)
        self._register("max", torch.zeros(self._input_dim))
        self._register("min", torch.zeros(self._input_dim))
        self._initialized = True
    # end _initialize

    # Aggregate information
    def _aggregate(self, x):
        """
        Aggregate information
        :param x: Input tensor
        """
        # Mean, std, mean length, max, min
        self._update_entry("mean", torch.mean(x, dim=0))
        self._update_entry("std", torch.std(x, dim=0))
        self._update_entry("mean_length", x.size(0))
        self._update_entry("max", torch.max(x, dim=0)[0])
        self._update_entry("min", torch.min(x, dim=0)[0])
    # end _aggregate

    # endregion OVERRIDE

# end StatsAggregator
