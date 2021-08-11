# -*- coding: utf-8 -*-
#
# File : echotorch/utils/visualisation/NodeObserver.py
# Description : Observe a Node object to visualise it activity afterward.
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
import torch
import networkx as nx
import matplotlib.pyplot as plt
from .Observable import Observable
from .ObservationPoint import ObservationPoint


# Observe a Observable object to visualise its activity afterward
class NodeObserver:
    """
    Observe an Observable object to visualise its activity afterward
    """

    # Constructor
    def __init__(self, node, initial_state=None, active=True):
        """
        Constructor
        :param node: Observable object to observe.
        :param initial_state: Observer's initial state
        :param active: Observing ?
        """
        # Save node
        self._node = node

        # Info on cell
        self._input_dim = node.input_dim
        self._hidden_dim = node.output_dim

        # From name to observation point
        self._observation_points = dict()
        self._observation_data = dict()

        # For each observation point
        for obp in node.observation_points:
            obp.register(self._observation_point_handler)
            self._observation_points[obp.name] = obp
            self._observation_data[obp] = dict()
        # end for

        # Current state
        self._current_state = initial_state

        # Active
        self._active = active
    # end __init__

    ##################
    # PUBLIC
    ##################

    # Set active/inactive
    def set_active(self, active_state):
        """
        Set active/inactive
        :param active_state: Active state
        """
        self._active = active_state
    # end set_active

    # Get active/inactive state
    def get_active(self):
        """
        Get active
        :return: True/False
        """
        return self._active
    # end get_active

    # Set state
    def set_state(self, new_state):
        """
        Set state of the observer
        :param new_state: State of the observer
        """
        self._current_state = new_state
    # end set_state

    # Get state
    def get_state(self):
        """
        Get state
        :return: Get current state of the observer
        """
        return self._current_state
    # end get_state

    # Draw matrix graph
    def draw_matrix_graph(self, matrix_name, draw=True, with_labels=True, font_weight='bold'):
        """
        Draw matrix graph
        :param matrix_name: Name of the ESNCell's parameter
        :return NetworkX graph
        """
        # Get matrix
        m = getattr(self._esn_cell, matrix_name)

        # M size
        m_dim = m.size(0)

        # New graph
        G = nx.Graph()

        # Add each nodes
        G.add_nodes_from(range(m.size(0)))

        # For each entry in m
        for i in range(m_dim):
            for j in range(m_dim):
                if m[i, j] != 0:
                    G.add_edge(i, j)
                # end if
            # end for
        # end for

        # Draw
        if draw:
            nx.draw(G, with_labels=with_labels, font_weight=font_weight)
        # end if

        return G
    # end draw_matrix_graph

    # Plot inputs
    def plot_inputs(self, sample_id, idxs, color, linewidth=1, start=0, length=-1, show_title=True, title="",
                    xticks=None, yticks=None, ylim=None, xlim=None):
        """
        Plot inputs
        :param sample_id: Index of the inputs to plot
        :param idxs: Indices of the inputs to plot
        :param start: Index of the starting point to plot
        :param length: Length of the plot
        """
        # Plot inputs
        if length == -1:
            plt.plot(self._esn_cell_inputs[sample_id][start:, idxs], color=color, linewidth=linewidth)
        else:
            plt.plot(self._esn_cell_inputs[sample_id][start:start + length, idxs], color=color, linewidth=linewidth)
        # end if

        # Title
        if show_title:
            plt.title(title)
        # end if

        # X labels
        if xlim is not None: plt.xlim(xlim)
        if xticks is not None: plt.xticks(xticks)

        # Y limits
        if ylim is not None: plt.ylim(ylim)
        if yticks is not None: plt.yticks(yticks)

    # end plot_inputs

    # Get data in the observer
    def get_data(self, point, states, idxs):
        """
        Get data in the observer
        :param point: Observation point (point name or NodeObserver object)
        :param states: Observer states to retrieve (state or list of state), or None for all
        :param idxs: An index (int), a list of indexes, a list of list (indexes for each states), or None for all
        :return: A list of tuple (data, forward index, sample index, time position)
        """
        # Get observation point
        if isinstance(point, ObservationPoint):
            point_obj = point
        else:
            point_obj = self._observation_points[point]
        # end if

        # List of observations
        list_of_observations = list()

        # For each observed states
        for state in self._observation_data[point_obj].keys():
            # List of data
            list_of_data = self._observation_data[point_obj][state]

            # State elligible for retrievale
            if (isinstance(states, str) and state == states) or (isinstance(states, list) and state in states) or states is None:
                # One simple index
                if isinstance(idxs, int) or isinstance(idxs, list):
                    list_of_observations.append((list_of_data[idxs], state, idxs))
                elif isinstance(idxs, list):
                    for idx in idxs:
                        list_of_observations.append((list_of_data[idx], state, idx))
                    # end for
                elif idxs is None:
                    for data_i in range(len(list_of_data)):
                        list_of_observations.append((list_of_data[data_i], state, data_i))
                    # end for
                # end if
            # end if
        # end for

        return list_of_observations
    # end get_data

    ##################
    # PRIVATE
    ##################

    # Observation point handler
    def _observation_point_handler(self, observation_point, data):
        """
        Observation point handler
        :param observation_point: Source observation point
        :param data: Data observed
        """
        # If active
        if self._active:
            # Create list for state if necessary
            if self._current_state not in self._observation_data[observation_point].keys():
                self._observation_data[observation_point][self._current_state] = list()
            # end if

            # Save data
            self._observation_data[observation_point][self._current_state].append(data)
        # end if
    # end _observation_point_handler

# end NodeObserver
