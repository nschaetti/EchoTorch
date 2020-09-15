# -*- coding: utf-8 -*-
#
# File : papers/gallicchio2017/tools/tools.py
# Description : Reproduction of the paper "Deep Reservoir Computing : A Critical Experiemental Analysis"
# (Gallicchio 2017)
# Date : 11th of September, 2020
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
# Copyright Nils Schaetti <nils.schaetti@unine.ch>, <nils.schaetti@unige.ch>

# Imports
import torch
from echotorch.utils import entropy


# Compute euclidian distances between states for a layer
def euclidian_distances(states1, states2, n_layers):
    """
    Compue euclidian distances between states for a layer
    :param states1:
    :param states2:
    :param n_layers
    :return: Tensor batch_size x time x n_layers
    """
    # Sizes
    batch_size = states1.size(0)
    time_length = states1.size(1)
    reservoir_size = int(states1.size(2) / n_layers)

    # Distances
    distances = torch.zeros(batch_size, time_length, n_layers)

    # For each batch
    for batch_i in range(batch_size):
        # For each layer
        for layer_i in range(n_layers):
            # Get layer states
            unperturbed_layer_states = states1[0, :, layer_i * reservoir_size:(layer_i + 1) * reservoir_size]
            perturbed_layer_states = states2[0, :, layer_i * reservoir_size:(layer_i + 1) * reservoir_size]

            # Compute Euclidian distance
            distances[batch_i, :, layer_i] = torch.sqrt(
                torch.sum(
                    torch.pow(
                        unperturbed_layer_states - perturbed_layer_states,
                        2
                    ),
                    dim=1
                )
            )
        # end for
    # end for

    return distances
# end euclidian_distances_layer


# Compute perturbation effect for each layer
# defined as P(l) = max_t(D(l, t) > 0)
def perturbation_effect(state_distances, e=1.0E-13):
    """
    Compute perturbation effect for each layer
    defined as P(l) = max_t(D(l, t) > 0)
    :param state_distances: Euclidian distances between states (batch_size, time_length, n_layers)
    :param e: Floor to be considered as zero
    :return: A tensor (batch_size, n_layers) with P(l) value for each layer l
    """
    # Sizes
    batch_size = state_distances.size(0)
    time_length = state_distances.size(1)
    n_layers = state_distances.size(2)

    # P
    P = torch.zeros(batch_size, n_layers)

    # For each batch
    for batch_i in range(batch_size):
        # For each layer
        for layer_i in range(n_layers):
            # Go through time
            for t in range(time_length):
                if state_distances[batch_i, t, layer_i] < e:
                    P[batch_i, layer_i] = t
                    break
                # end if
            # end for
        # end for
    # end for

    return P
# end perturbation_effect


# Rank layers by the length of the perturbation effect
def ranking_of_layers(P):
    """
    Rank layers by the length of the perturbation effect
    :param P: Pertubation effects (batch size x n_layers)
    :return: A tensor (batch size x n_layers)
    """
    # Sizes
    batch_size = P.size(0)
    n_layers = P.size(1)

    # Ranking
    ranking = torch.zeros(batch_size, n_layers)

    # For each batch
    for batch_i in range(batch_size):
        # Tuple with (layer index, perturbation effect)
        pertur_effects = [(layer_i, P[batch_i, layer_i]) for layer_i in range(n_layers)]

        # Rank from smallest effect to biggest
        pertur_effects = sorted(pertur_effects, key=lambda tup: tup[1])

        # Set
        ranking[batch_i] = torch.tensor([x[0] for x in pertur_effects])
    # end for

    return ranking
# end ranking_of_layers


# Kendall's tau
# Qualitative measure
def kendalls_tau(ranking):
    """
    Kandall's tau
    :param ranking: Tensor of rankings (batch size x n_layers)
    :return: List of Kandall's taus
    """
    # Sizes
    batch_size = ranking.size(0)
    n_layers = ranking.size(1)

    # List of taus
    taus = list()

    # For each batch
    for batch_i in range(batch_size):
        # Count the number of pair swap
        pair_swaps = 0

        # For each combinaison of layers
        for layer_1 in range(n_layers):
            for layer_2 in range(n_layers):
                if layer_1 != layer_2:
                    if layer_1 < layer_2 and ranking[batch_i, layer_1] > ranking[batch_i, layer_2]:
                        pair_swaps += 1
                    # end if
                # end for
            # end for
        # end for

        # Append
        taus.append(pair_swaps)
    # end for

    return taus
# end kendalls_tau


# Spearman's rule
# Qualitative measure
def spearmans_rule(ranking):
    """
    Spearman's Rule
    :param ranking: Tensor of rankings (batch size x n_layers)
    :return: List of Spearman's values
    """
    # Sizes
    batch_size = ranking.size(0)
    n_layers = ranking.size(1)

    # List of taus
    spearmans = list()

    # For each batch
    for batch_i in range(batch_size):
        # Spearman value
        SV = 0

        # For each layer
        for layer_i in range(n_layers):
            SV += abs(layer_i - ranking[batch_i, layer_i].item())
        # end for

        # Append
        spearmans.append(SV)
    # end for

    return spearmans
# end spearmans_rule


# Time-scales separation
# Quantitative measure
def timescales_separation(P):
    """
    Time-scales separation
    :param P: Pertubation effects (batch size x n_layers)
    :return:
    """
    # Sizes
    batch_size = P.size(0)
    n_layers = P.size(1)

    # Save separation for each element in batch
    TS_separation = list()

    # For each batch
    for batch_i in range(batch_size):
        # Interstate distance between duration of perturbation
        IS = 0

        # For each layer
        for layer_i in range(1, n_layers):
            IS += P[batch_i, layer_i].item() - P[batch_i, layer_i-1].item()
        # end for

        # Append
        TS_separation.append(IS)
    # end for

    return TS_separation
# end timescales_separation


# Entropy per layer
def entropy_layer(reservoir_states, n_layers):
    """
    Entropy per layer
    :param reservoir_states: Reservoir states (batch size, time length, reservoir size)
    :param n_layers: Number of layers
    :return: Entropy per layer
    """
    # Dim. sizes
    batch_size = reservoir_states.size(0)
    time_length = reservoir_states.size(1)
    total_reservoir_size = reservoir_states.size(2)

    # Reservoir size
    reservoir_size = int(total_reservoir_size / n_layers)

    # Switch time and units dimensions
    reservoir_states = reservoir_states.transpose(0, 2, 1)

    # Tensor to save entropy per layer
    entropy_per_layer = torch.zeros(n_layers)

    # For each layer
    for layer_i in range(n_layers):
        entropy_per_layer[layer_i] = entropy(reservoir_states[:, layer_i*reservoir_size:(layer_i+1)*reservoir_size, :])
    # end for

    return entropy_per_layer
# end entropy_layer
