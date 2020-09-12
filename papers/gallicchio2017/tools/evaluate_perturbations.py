# -*- coding: utf-8 -*-
#
# File : papers/gallicchio2017/tools/evaluate_perturbations.py
# Description : Reproduction of the paper "Deep Reservoir Computing : A Critical Experiemental Analysis"
# (Gallicchio 2017)
# Date : 10th of September, 2020
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
import torch.utils.data
import echotorch.nn as etnn
import echotorch.datasets as etds
import echotorch.transforms as ettr
from echotorch.utils.matrix_generation import matrix_factory
from torch.autograd import Variable
from papers.gallicchio2017.tools import euclidian_distances, perturbation_effect, ranking_of_layers, kendalls_tau, \
    spearmans_rule, timescales_separation


# Evaluate perturbation experiement
def evaluate_perturbations(n_layers, reservoir_size, w_connectivity, win_connectivity, leak_rate, spectral_radius,
                           vocabulary_size, input_scaling, bias_scaling, esn_type, n_samples, sample_len,
                           perturbation_position, use_cuda, dtype):
    """
    Evaluate perturbation experiment
    """
    # Dataset generating sequences of random symbols
    random_symbols_loader = torch.utils.data.DataLoader(
        etds.TransformDataset(
            root_dataset=etds.RandomSymbolDataset(
                sample_len=sample_len,
                n_samples=n_samples,
                vocabulary_size=10
            ),
            transform=ettr.timeseries.ToOneHot(output_dim=vocabulary_size, dtype=dtype),
            transform_indices=[0]
        )
    )

    # Generator for internal matrices W
    w_generator = matrix_factory.get_generator(
        name='normal',
        spectral_radius=spectral_radius,
        connectivity=w_connectivity
    )

    # Generator for inputs-to-reservoir matrices Win
    win_generator = matrix_factory.get_generator(
        name='normal',
        connectivity=win_connectivity
    )

    # Generator for internal units biases
    wbias_generator = matrix_factory.get_generator(
        name='normal',
        scale=bias_scaling
    )

    # Deep ESN / ESN
    if esn_type == 'esn':
        esn_model = etnn.reservoir.LiESN(
            input_dim=vocabulary_size,
            hidden_dim=reservoir_size,
            output_dim=vocabulary_size,
            leaky_rate=leak_rate,
            w_generator=w_generator,
            win_generator=win_generator,
            wbias_generator=wbias_generator,
            input_scaling=input_scaling,
            dtype=dtype
        )
    else:
        esn_model = etnn.reservoir.DeepESN(
            n_layers=n_layers,
            input_dim=vocabulary_size,
            hidden_dim=reservoir_size,
            output_dim=vocabulary_size,
            leak_rate=leak_rate,
            w_generator=w_generator,
            win_generator=win_generator,
            wbias_generator=wbias_generator,
            input_scaling=input_scaling,
            input_type=esn_type,
            dtype=dtype
        )
    # end if

    # Use cuda ?
    if use_cuda:
        esn_model.cuda()
    # end if

    # Return per batch
    return_list = list()

    # Go through all the dataset
    for batch_idx, data in enumerate(random_symbols_loader):
        # Data
        unperturbed_input = data[0]

        # Copy the unperturbed sequence
        perturbed_input = unperturbed_input.clone()

        # Introduce a perturbation at t=perturbation_position for each batch
        for batch_i in range(perturbed_input.size(0)):
            # Current input
            current_input = perturbed_input[batch_i, perturbation_position]

            # Changed
            changed = False

            # Try until it is new
            while not changed:
                # Change input at time t
                perturbed_input[batch_i, perturbation_position] = torch.zeros(vocabulary_size)
                perturbed_input[batch_i, perturbation_position, torch.randint(vocabulary_size, size=(1, 1)).item()] = 1.0

                # Changed ?
                changed = not torch.eq(perturbed_input[batch_i, perturbation_position], current_input)
            # end while
        # end for

        # To variables
        unperturbed_input, perturbed_input = Variable(unperturbed_input.double()), Variable(perturbed_input.double())

        # CUDA
        if use_cuda:
            unperturbed_input = unperturbed_input.cuda()
            perturbed_input = perturbed_input.cuda()
        # end if

        # Feed both version to the ESN/DeepESN
        unperturbed_states = esn_model(unperturbed_input, unperturbed_input)
        perturbed_states = esn_model(perturbed_input, perturbed_input)

        # Compute euclidian distance between each state for each layer
        states_distances = euclidian_distances(
            unperturbed_states,
            perturbed_states,
            n_layers
        )

        # Distances after the perturbation
        states_distances = states_distances[:, perturbation_position:, :]

        # Perturbation effect
        P = perturbation_effect(states_distances)

        # Compute ranking
        layer_ranking = ranking_of_layers(P)

        # Compute Kendall's tau
        tau = kendalls_tau(ranking=layer_ranking)

        # Compute Spearman's rule
        sp_rule = spearmans_rule(ranking=layer_ranking)

        # Compute timescales separation
        ts_separation = timescales_separation(P)

        # Add to list
        return_list.append((states_distances, P, layer_ranking, tau, sp_rule, ts_separation))
    # end for

    return return_list
# end evaluate_perturbations
