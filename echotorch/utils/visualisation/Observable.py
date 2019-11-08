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


# Define a class of object with observable properties for visualisation
class Observable:
    """
    Define a class of object observable for visualisation
    """

    # Constructor
    def __init__(self, observe_hooks):
        """
        Constructor
        """
        # Keep the name of hooks
        self._observe_hooks = observe_hooks

        # We save handlers for each hook
        self._observe_handlers = dict()
        for hook in observe_hooks:
            self._observe_handlers[hook] = list()
        # end for
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
        return self._observe_hooks
    # end observation_points

    ###############
    # PUBLIC
    ###############

    # Add an handler for observation
    def observe(self, hook, handler):
        """
        Add an handler for observation
        :param hook: The name of the hook to observe
        :param handler: The function to call
        """
        if hook in self._observe_hooks:
            self._observe_handlers[hook] = handler
        else:
            raise Exception("No observation hook named {}".format(hook))
        # end if
    # end observe

    ###############
    # PRIVATE
    ###############

    # Send item for observation point
    def observation_point(self, hook_name, item, batch, sample, t):
        """
        Send item for observation point
        :param hook_name: The name of the observation hook
        :param item: Item to send
        :param batch: Batch index
        :param sample: Sample index
        :param t: time position
        """
        # For each handlers
        for handler in self._observe_handlers[hook_name]:
            handler(item, batch, sample, t)
        # end for
    # end observation_point

# end Observable
