# -*- coding: utf-8 -*-
#
# File : echotorch/utils/visualisation/Visualiser.py
# Description : Base class for Visualiser.
# Date : 29th of November, 2019
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


# Base class for Visualiser
class Visualiser:
    """
    Base class for Visualiser
    """

    # Constructor
    def __init__(self, observer):
        """
        Constructor
        :param observer: Observer to use for visualisation
        """
        self._observer = observer
    # end __init__

    ################
    # PROPERTIES
    ################

    ################
    # PUBLIC
    ################

    ################
    # PRIVATE
    ################

    # Get observer data
    def _get_observer_data(self, point_name, states, idxs, concat=False, concat_dim=0):
        """
        Get observer data
        :param point_name: Observation point name
        :param states: Observer states
        :param idxs: Sample indices in the state collection
        :param concat: Concat data ?
        :param concat_dim: Concat data on which dimension
        :return: Observed data
        """
        # Get data
        cell_observations = self._observer.get_data(point_name, states, idxs)

        # No-concat
        if not concat:
            # List of data
            list_of_data = list()

            # For each observed data
            for data, data_state, data_i in cell_observations:
                list_of_data.append(data)
            # end for

            return list_of_data
        else:
            concat_data = None
            # For each observed data
            for data, data_state, data_i in cell_observations:
                if data_i == 0:
                    concat_data = data
                else:
                    concat_data = torch.cat((concat_data, data), dim=concat_dim)
                # end if
            # end for
            return concat_data
        # end if
    # end _get_observer_data

# end Visualiser
