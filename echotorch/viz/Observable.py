# -*- coding: utf-8 -*-
#
# File : echotorch/utils/visualisation/Observable.py
# Description : Define a class of object with observable properties for visualisation.
# Date : 7th of November, 2019
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


# Import
from .ObservationPoint import ObservationPoint


# Define a class of object with observable properties for visualisation
class Observable:
    """
    Define a class of object observable for visualisation
    """

    # Constructor
    def __init__(self):
        """
        Constructor
        """
        # Keep the name of hooks
        self._observation_points = list()
    # end __init__

    ###############
    # STATIC
    ###############

    # Observation points
    @property
    def observation_points(self):
        """
        Observation point
        :return: List of observation points (string)
        """
        return self._observation_points
    # end observation_points

    ###############
    # PUBLIC
    ###############

    # Add an handler for observation
    def observe(self, point_name, handler):
        """
        Add an handler for observation
        :param point_name: The name of the point to observe
        :param handler: The function to call
        """
        # Point found
        point_found = False

        # For each point
        for point in self.observation_points:
            if point.name == point_name:
                point.register(handler)
                point_found = True
            # end if
        # end for

        if not point_found:
            raise Exception("No observation point named {}".format(point))
        # end if
    # end observe

    # Send item for observation point
    def observation_point(self, point_name, data):
        """
        Send item for observation point
        :param point_name: The name of the observation hook
        :param data: Observed data
        """
        # For each handlers
        for point in self.observation_points:
            if point.name == point_name:
                point(data)
            # end if
        # end for
    # end observation_point

    # Add observation point(
    def add_observation_point(self, name, unique):
        """
        Add observation point
        :param name: Point name
        :param unique: The point is called one-time ?
        """
        self._observation_points.append(ObservationPoint(name=name, unique=unique))
    # end _add_observation_point

    ###############
    # PRIVATE
    ###############

# end Observable
