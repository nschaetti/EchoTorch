# -*- coding: utf-8 -*-
#
# File : echotorch/transforms/Aggregator.py
# Description : Aggregators are used to compute statistics and informations about timeseries.
# Date : 18th of January, 2021
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
from .Transformer import Transformer


# Base class for aggregators
class Aggregator(Transformer):
    """
    Base class for aggregators
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, input_dim, dtype=torch.float64):
        """
        Constructor
        :param input_dim: Input data dimension
        :param dtype: Data type
        """
        # Super
        super(Aggregator, self).__init__(input_dim, 0, dtype)

        # Dictionary for data
        self._aggregated_data = dict()
        self._aggregated_count = dict()

        # Flags
        self._is_initialized = False
        self._is_finalized = False

        # Initialize aggregator
        self._initialize()
    # end __init__

    # endregion CONSTRUCTORS

    # region PUBLIC

    # Is initialized?
    def is_initialized(self) -> bool:
        """
        Is the aggregator initialized?
        :return: True or False
        """
        return self._is_initialized
    # end is_initialized

    # Is finalized?
    def is_finalized(self) -> bool:
        """
        Is the aggregator finalized?
        :return: True or False
        """
        return self._is_finalized
    # end is_finalized

    # endregion PUBLIC

    # region PRIVATE

    # Update entry
    def _update_entry(self, entry_name, entry_value, inc=True):
        """
        Update an entry in the registry
        :param entry_name: Entry name
        :param entry_value: New entry value
        :param inc: Increment counter?
        """
        self._aggregated_data[entry_name] = entry_value
        if inc:
            self._aggregated_count[entry_name] += 1.0
        # end if
    # end _update_entry

    # Create an entry
    def _create_entry(self, entry_name, entry_value=0, create_counter=True):
        """
        Create an entry in the registry
        :param entry_name: Entry name
        :param entry_value: Entry value
        :param create_counter: Create entry in counters?
        """
        self._aggregated_data[entry_name] = entry_value
        if create_counter:
            self._aggregated_count[entry_name] = 0.0
        # end if
    # end _create_entry

    # Get counter
    def _get_counter(self, entry_name):
        """
        Get counter
        :param entry_name: Entry name
        :return: Counter for this entry
        """
        return self._aggregated_count[entry_name]
    # end _get_counter

    # endregion PRIVATE

    # region OVERRIDE

    # Transform to aggregate
    def _transform(self, x):
        """
        Transform to aggregate
        :param x: Input data tensor
        :return: None
        """
        self._aggregate(x)
    # end _transform

    # Get item
    def __getitem__(self, item):
        """
        Get item (in the registry)
        :param item: Entry name
        :return: Entry value
        """
        return self._aggregated_data[item]
    # end __getitem__

    # endregion OVERRIDE

    # region TO_IMPLEMENT

    # Init. the aggregator
    def _initialize(self):
        """
        Init the aggregator
        """
        raise Exception("Initialization function must be implemented")
    # end _init

    # Aggregate information
    def _aggregate(self, x):
        """
        Aggregate information
        :param x: Input data tensor
        """
        raise Exception("Aggregate function must be implemented")
    # end _aggregate

    # Finalize aggregation
    def _finalize(self):
        """
        Finalize aggregation
        """
        raise Exception("Finalize aggregation")
    # end _finalize

    # endregion TO_IMPLEMENT

# end Aggregator
