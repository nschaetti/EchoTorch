# -*- coding: utf-8 -*-
#
# File : echotorch/transforms/Aggregator.py
# Description : A special type of Transformer which compute only statistics based on data.
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
from .Transformer import Transformer


# Aggregator
class Aggregator(Transformer):
    """
    A special type of Transformer which compute only statistics based on data.
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, input_dim, time_dim=0, dtype=torch.float64):
        """
        Constructor
        """
        # Super constructor
        super(Aggregator, self).__init__(input_dim, input_dim, time_dim, dtype)

        # Data gathering
        self._data = dict()
        self._counters = dict()

        # State
        self._initialized = False

        # Initialize
        self._initialize()
    # end __init__

    # endregion CONSTRUCTORS

    # region PROPERTIES

    # Data
    @property
    def data(self):
        """
        Data
        """
        return self._data
    # end data

    # Counter
    @property
    def counters(self):
        """
        Counters
        """
        return self._counters
    # end counters

    # Registered entries
    @property
    def entries(self):
        """
        Registered entries
        """
        return self._data.keys()
    # end entries

    # Is initialized
    @property
    def initialized(self):
        """
        Initialized
        """
        return self._initialized
    # end initialized

    # endregion PROPERTIES

    # region PUBLIC

    # Counters
    def counter(self, entry_name):
        """
        Counter
        """
        return self._counters[entry_name]
    # end counter

    # endregion PUBLIC

    # region PRIVATE

    # Register an entry
    def _register(self, entry_name, initial_value):
        """
        Register an entry
        """
        self._data[entry_name] = initial_value
        self._counters[entry_name] = 0
    # end _register

    # Increment counter
    def _inc(self, entry_name):
        """
        Increment counter
        """
        self._counters[entry_name] += 1
        # print("_inc: {}".format(self._counters))
    # end _inc

    # endregion PRIVATE

    # region OVERRIDE

    # Transform
    def _transform(self, x):
        """
        Transform input
        """
        self._aggregate(x)
        return x
    # end _transform

    # Get item
    def __getitem__(self, item):
        """
        Get item
        """
        return self._data[item], self._counters[item]
    # end __getitem__

    # Set item
    def __setitem__(self, key, value):
        """
        Set item
        """
        self._data[key] = value
    # end __setitem__

    # endregion OVERRIDE

    # region TO_IMPLEMENT

    # Initialize the aggregator
    def _initialize(self):
        """
        Initialize the aggregator
        """
        raise Exception("Initialize not implemented")
    # end _initialize

    # Aggregate information
    def _aggregate(self, x):
        """
        Aggregate information
        """
        pass
    # end aggregate

    # endregion TO_IMPLEMENT

# end Aggregator

