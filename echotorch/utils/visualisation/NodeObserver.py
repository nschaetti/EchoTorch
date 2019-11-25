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


# Observe a Observable object to visualise its activity afterward
class NodeObserver:
    """
    Observe an Observable object to visualise its activity afterward
    """

    # Constructor
    def __init__(self, node):
        """
        Constructor
        :param node: Observable object to observe.
        """
        # Save node
        self._node = node

        # Info on cell
        self._input_dim = node.input_dim
        self._hidden_dim = node.output_dim

        # For each observation point
        for obp in node.observation_points:
            obp.register(self._observation_point_handler)
        # end for
    # end __init__

    ##################
    # PUBLIC
    ##################

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

    # Plot neurons activities
    def plot_neurons(self, sample_id, idxs, color, linewidth=1, start=0, length=-1, show_title=True, title="",
                     xticks=None, yticks=None, ylim=None, xlim=None):
        """
        Plot neuron activities
        :param sample_id: Index of the sample to plot
        :param idxs: Indices of the neurons to plot
        :param start: Index of the starting point to plot
        :param length: Length of the plot
        :param show_title:
        :param title:
        :param xticks:
        :param yticks:
        :param ylim:
        :param xlim:
        """
        # Plot neurons
        if length == -1:
            plt.plot(self._esn_cell_states[sample_id][start:, idxs], color=color, linewidth=linewidth)
        else:
            plt.plot(self._esn_cell_states[sample_id][start:start + length, idxs], color=color, linewidth=linewidth)
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
    # end plot_neurons

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

    # Plot states singular values
    def plot_state_singular_values(self, sample_id, color, linewidth=1, length=-1, show_title=True, title="",
                                   xticks=None, yticks=None, ylim=None, xlim=None, log10=False):
        """
        Plot states singular values
        :param sample_id:
        :param color:
        :param linewidth:
        :param ylim:
        :param length:
        :param show_title:
        :param title:
        :param log10:
        :return:
        """
        # State sample
        state_sample = self._esn_cell_states[sample_id]

        # Correlation matrix R
        R = torch.mm(state_sample.t(), state_sample) / state_sample.size(0)

        # SVD on state
        _, S, _ = torch.svd(R)

        # Log10?
        if log10:
            S = torch.log10(S)
        # end if

        # Learning PC energy
        if length != -1:
            plt.plot(S[:length], color=color, linewidth=linewidth)
        else:
            plt.plot(S, color=color, linewidth=linewidth)
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
    # end plot_state_singular_values

    ##################
    # PRIVATE
    ##################

    # Observation point handler
    def _observation_point_handler(self, observation_point, data, batch_i, sample_i, t):
        """
        Observation point handler
        :param observation_point: Source observation point
        :param data: Data observed
        :param batch_i: Batch index (Call to forward)
        :param sample_i: Sample index
        :param t: Timestep
        """
        pass
    # end

# end NodeObserver
