# -*- coding: utf-8 -*-
#
# File : echotorch/utils/visualisation/ObservationPoint.py
# Description : A point of observation for the Observable object.
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


# Imports


# A point of observation for Observable objects
class ObservationPoint:
    """
    A point of observation for Observable objects
    """

    # Constructor
    def __init__(self, unique, input_dim, batch_dim, time_dim):
        """
        Constructor
        :param unique: Data is set only one time (model's parameter for example)
        :param input_dim: Input dimension (of observed data)
        :param batch_dim: Batch dimension (-1 if no batch)
        :param time_dim: Time dimension (-1 if timeless data)
        """
        # Properties
        self._unique = unique
        self._input_dim = input_dim
        self._batch_size = batch_dim
        self._time_dim = time_dim

        # Handlers
        self._handlers = list()
    # end __init__

    #################
    # PROPERTIES
    #################

    # Unique data ?
    @property
    def unique(self):
        """
        Unique data (set only one time)
        :return: Uniqueness
        """
        return self._unique
    # end unique

    ################
    # PUBLIC
    ################

    # Register to this point
    def register(self, handle_func):
        """
        Register to this point
        :param handle_func: Function call when this point is observed
        """
        if handle_func not in self._handlers:
            self._handlers.append(handle_func)
        # end if
    # end register

    ################
    # PRIVATE
    ################

    ################
    # OVERRIDE
    ################

    # Call the observation point
    def __call__(self, *args, **kwargs):
        """
        Call the observation point
        :param args:
        :param kwargs:
        :return:
        """
        pass
    # end __call__

# end ObservablePoint
