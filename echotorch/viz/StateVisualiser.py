# -*- coding: utf-8 -*-
#
# File : echotorch/utils/visualisation/StateVisualiser.py
# Description : Visualise reservoir states.
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
import networkx as nx
import matplotlib.pyplot as plt
from .Visualiser import Visualiser
from echotorch.utils.utility_functions import generalized_squared_cosine


# Visualise reservoir states
class StateVisualiser(Visualiser):
    """
    Visualise reservoir states
    """

    ################
    # PROPERTIES
    ################

    ################
    # PUBLIC
    ################

    # Compute R-based similarity matrix
    def compute_R_similarity_matrix(self, point_name, states, idxs, sim_func=generalized_squared_cosine):
        """
        Compute R-based similarity matrix
        :param point_name: Observation point
        :param states: List of states (or list of list) used to compute similarities
        :param idxs: Sample indices in state collection
        :param sim_func: Similarity function (default: generalized_squared_cosine)
        :return: Similarity matrix as torch Tensor
        """
        # Cell states
        cell_states = list()

        # For each states
        for state in states:
            # Get observed data
            cell_states.append(self._get_observer_data(point_name, state, idxs, concat=True, concat_dim=2))
        # end for

        # Similarity matrix
        sim_matrix = torch.zeros(len(cell_states), len(cell_states))

        # For each pair
        for i in range(len(cell_states)):
            for j in range(len(cell_states)):
                # R matrices
                R_i = torch.mm(cell_states[i].t(), cell_states[i])
                R_j = torch.mm(cell_states[j].t(), cell_states[j])

                # SVDs
                Ui, Si, _ = torch.svd(R_i)
                Uj, Sj, _ = torch.svd(R_j)

                # Similarity matrix
                sim_matrix[i, j] = sim_func(Si, Ui, Sj, Uj)
            # end for
        # end for

        return sim_matrix
    # end compute_R_similarity_matrix

    # Plot neurons activities
    def plot_neurons(self, point_name, states, idxs, neuron_idxs, colors, linewidth=1, start=0, length=-1,
                     show_title=True, title="", xticks=None, yticks=None, ylim=None, xlim=None):
        """
        Plot neuronal activities
        :param point_name: Source observation point
        :param states: Observer's states
        :param idxs: Sample indices in the state collection
        :param neuron_idxs: Indices of the neurons to plot
        :param linewidth: Line width
        :param colors: Line color
        :param start: Index of the starting point to plot
        :param length: Length of the plot
        :param show_title: Add title to the plot ?
        :param title: Plot's title
        :param xticks: X-ticks
        :param yticks: Y-ticks
        :param ylim: Y limits
        :param xlim: X limits
        """
        # Get observed data
        cell_states = self._get_observer_data(point_name, states, idxs, concat=True, concat_dim=2)

        # Plot neurons
        for neuron_i, neuron_idx in enumerate(neuron_idxs):
            n_color = colors[neuron_i] if type(colors) == list else colors
            if length == -1:
                plt.plot(cell_states[start:, neuron_idx], color=n_color, linewidth=linewidth)
            else:
                plt.plot(cell_states[start:start + length, neuron_idx], color=n_color, linewidth=linewidth)
            # end if
        # end for

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

    # Plot states singular values
    def plot_singular_values(self, point_name, states, idxs, color, linewidth=1, length=-1, show_title=True, title="",
                                   xticks=None, yticks=None, ylim=None, xlim=None, log10=False):
        """
        Plot singular values
        :param point_name: Name of the observation point
        :param states: Observer's state (or None if all)
        :param idxs: Samples indices in the state collection (or None if all)
        :param color: Line color
        :param linewidth: Line width
        :param length: Length of the plot
        :param show_title: Add title to the plot ?
        :param title: Plot title
        :param xticks: X ticks
        :param yticks: Y ticks
        :param ylim: Y limits [min, max]
        :param xlim: X limits [min, max]
        :param log10: If true, show log10 of singular values
        """
        # Get observed data
        cell_states = self._get_observer_data(point_name, states, idxs, concat=True, concat_dim=2)

        # Correlation matrix R
        R = torch.mm(cell_states.t(), cell_states) / cell_states.size(0)

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

# end StateVisualiser
