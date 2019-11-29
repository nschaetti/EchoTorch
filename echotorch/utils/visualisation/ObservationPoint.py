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
    def __init__(self, name, unique):
        """
        Constructor
        :param name: Point's name
        :param unique: Data is set only one time (model's parameter for example)
        """
        # Properties
        self._name = name
        self._unique = unique

        # Handlers
        self._handlers = list()
    # end __init__

    #################
    # PROPERTIES
    #################

    # Point's name
    @property
    def name(self):
        """
        Point's name
        :return: Point's name
        """
        return self._name
    # end name

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
    def __call__(self, data):
        """
        Call the observation point
        :param data: Observed data
        """
        # For each handling function
        for handle_func in self._handlers:
            handle_func(self, data)
        # end for
    # end __call__

# end ObservablePoint
