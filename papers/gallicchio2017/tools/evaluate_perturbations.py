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
import numpy as np
import echotorch.nn as etnn
import echotorch.datasets as etds
import echotorch.transforms as ettr
from echotorch.utils.matrix_generation import matrix_factory
from torch.autograd import Variable
from .tools import euclidian_distances, perturbation_effect, ranking_of_layers, kendalls_tau, \
    spearmans_rule, timescales_separation


# Get hyperparameter value
def get_hyperparam_value(hyperparam, layer_i):
    """
    Get hyperparameter value
    :param hyperparam: Hyperparameter (a value or a list)
    :param layer_i: Which layer (integer)
    :return: Hyperparameter value for this layer
    """
    if type(hyperparam) == list or type(hyperparam) == np.ndarray or type(hyperparam) == torch.tensor:
        return hyperparam[layer_i]
    else:
        return hyperparam
    # end if
# end get_hyperparam_value


# Evaluate perturbation experiement
def evaluate_perturbations(n_layers, reservoir_size, w_connectivity, win_connectivity, leak_rate, spectral_radius,
                           vocabulary_size, input_scaling, bias_scaling, esn_type, n_samples, sample_len,
                           perturbation_position, dataset=None, use_cuda=False, dtype=torch.float64):
    """
    Evaluate perturbation experiment
    """
    # Dataset generating sequences of random symbols
    if dataset is None:
        random_symbols_loader = torch.utils.data.DataLoader(
            etds.TransformDataset(
                root_dataset=etds.RandomSymbolDataset(
                    sample_len=sample_len,
                    n_samples=n_samples,
                    vocabulary_size=10
                ),
                transform=ettr.timeseries.ToOneHot(output_dim=vocabulary_size, dtype=dtype),
                transform_indices=None
            )
        )
    else:
        random_symbols_loader = torch.utils.data.DataLoader(dataset)
    # end if

    # List of generator
    w_generators = list()
    win_generators = list()
    wbias_generators = list()

    # For each layer
    for layer_i in range(n_layers):
        # Generator for internal matrices W
        w_generators.append(matrix_factory.get_generator(
            name='normal',
            spectral_radius=get_hyperparam_value(spectral_radius, layer_i),
            connectivity=get_hyperparam_value(w_connectivity, layer_i)
        ))

        # Generator for inputs-to-reservoir matrices Win
        win_generators.append(matrix_factory.get_generator(
            name='normal',
            connectivity=get_hyperparam_value(win_connectivity, layer_i)
        ))

        # Generator for internal units biases
        wbias_generators.append(matrix_factory.get_generator(
            name='normal',
            scale=get_hyperparam_value(bias_scaling, layer_i)
        ))
    # end for

    # Deep ESN / ESN
    if esn_type == 'esn':
        esn_model = etnn.reservoir.LiESN(
            input_dim=vocabulary_size,
            hidden_dim=reservoir_size,
            output_dim=vocabulary_size,
            leaky_rate=leak_rate,
            w_generator=w_generators[0],
            win_generator=win_generators[0],
            wbias_generator=wbias_generators[0],
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
            w_generator=w_generators,
            win_generator=win_generators,
            wbias_generator=wbias_generators,
            input_scaling=input_scaling,
            input_type=esn_type,
            dtype=dtype
        )
    # end if

    # Use cuda ?
    if use_cuda:
        esn_model.cuda()
    # end if

    # Variable to keep the average over samples
    average_state_distances = torch.zeros(sample_len - perturbation_position, n_layers)
    average_KT = 0.0
    average_SF = 0.0
    average_TS = 0.0
    average_count = 0

    # Go through all the dataset
    for batch_idx, data in enumerate(random_symbols_loader):
        # Data
        unperturbed_input = data

        # Copy the unperturbed sequence
        perturbed_input = unperturbed_input.clone()

        # Introduce a perturbation at t=perturbation_position for each batch
        for batch_i in range(perturbed_input.size(0)):
            # Current input
            current_input = perturbed_input[batch_i, perturbation_position].clone()

            # Changed
            changed = False

            # Try until it is new
            while not changed:
                # Change input at time t
                perturbed_input[batch_i, perturbation_position] = torch.zeros(vocabulary_size)
                perturbed_input[batch_i, perturbation_position, torch.randint(vocabulary_size, size=(1, 1)).item()] = 1.0

                # Changed ?
                changed = not torch.all(torch.eq(perturbed_input[batch_i, perturbation_position], current_input))
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
        # state_distances is (batch size, time length, n. layers)
        states_distances = euclidian_distances(
            unperturbed_states,
            perturbed_states,
            n_layers
        )

        # Keep only distances after the perturbation
        states_distances = states_distances[:, perturbation_position:, :]

        # Perturbation effect
        # P is (batch size, n. layers)
        P = perturbation_effect(states_distances)

        # Compute ranking
        # layer_ranking is (batch size, n. layers)
        layer_ranking = ranking_of_layers(P)

        # Compute Kendall's tau
        # KT is (batch size)
        KT = kendalls_tau(ranking=layer_ranking)

        # Compute Spearman's rule
        # SF is (batch size)
        SF = spearmans_rule(ranking=layer_ranking)

        # Compute timescales separation
        # TS is (batch size)
        TS = timescales_separation(P)

        # Add to averaging variables
        for batch_i in range(perturbed_input.size(0)):
            average_state_distances += states_distances[0]
            average_KT += KT[0]
            average_SF += SF[0]
            average_TS += TS[0]
            average_count += 1
        # end for
    # end for

    # Average
    average_state_distances /= average_count
    average_KT /= average_count
    average_SF /= average_count
    average_TS /= average_count

    return esn_model, average_state_distances, average_KT, average_SF, average_TS
# end evaluate_perturbations
