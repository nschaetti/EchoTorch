# -*- coding: utf-8 -*-
#
# File : papers/gallicchio2017/deep_reservoir_hyperparameters_search.py
# Description : Reproduction of the paper "Deep Reservoir Computing : A Critical Experiemental Analysis"
# (Gallicchio 2017)
# Date : 14th of September, 2020
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
import echotorch.utils
import echotorch.datasets as etds
import echotorch.transforms as ettr
import numpy as np
import echotorch.utils.optimization as optim
from papers.gallicchio2017.tools import evaluate_perturbations


# Function to test the ESN with specific hyper-parameters
def evaluation_function(parameters, datasets, n_layers, reservoir_size, esn_type,
                        samples_per_model, vocabulary_size=10, sample_len=5000, perturbation_position=100,
                        n_samples=10, fitness_measure='KT', use_cuda=False, dtype=torch.float64):
    """
    Function to test the ESN with specific hyper-parameters
    :param parameters: Possible parameters values for to optimize
    :param datasets: Dataset to use
    :param n_samples: Number of samples to evaluate a set of parameters
    :param n_layers: Number of layers in the DeepESN
    :param reservoir_size: Size of each reservoir
    :param vocabulary_size: Size of the vocabulary
    :param esn_type: Type of ESN
    :param samples_per_model: How many samples in the dataset
    :param sample_len: Length of the sample
    :param perturbation_position: Position of the perturbation
    :return:
    """
    # Get hyperparameters
    w_connectivity = parameters['w_connectivity']
    win_connectivity = parameters['win_connectivity']
    leak_rate = parameters['leak_rate']
    spectral_radius = parameters['spectral_radius']
    input_scaling = parameters['input_scaling']
    bias_scaling = parameters['bias_scaling']

    # Average fitness value
    average_fitness = 0.0
    average_count = 0

    # Perform experiment with the model
    for sample_i in range(n_samples):
        # Evaluate perturbations
        esn_model, states_distances, KT, SF, TS = evaluate_perturbations(
            n_layers=n_layers,
            reservoir_size=reservoir_size,
            w_connectivity=w_connectivity,
            win_connectivity=win_connectivity,
            leak_rate=leak_rate,
            spectral_radius=spectral_radius,
            vocabulary_size=vocabulary_size,
            input_scaling=input_scaling,
            bias_scaling=bias_scaling,
            esn_type=esn_type,
            n_samples=samples_per_model,
            sample_len=sample_len,
            perturbation_position=perturbation_position,
            dataset=datasets,
            use_cuda=use_cuda,
            dtype=dtype
        )

        # Add to fitness measure
        if fitness_measure == 'KT':
            average_fitness += KT
        elif fitness_measure == 'SF':
            average_fitness += SF
        elif fitness_measure == 'TS':
            average_fitness += TS
        # end if

        # Count
        average_count += 1
    # end for

    # Average
    average_fitness /= average_count

    return esn_model, average_fitness
# end evaluation_function


# Generate a random array for a parameter, with a value for each layer
def generate_random_parameter_value(mini, maxi, num_layers, num_chromosomes, logspace=False, base=10):
    """
    Generate a random array for a parameter, with a value for each layer
    :param mini: Minimum values for the parameter
    :param maxi: Maximum values for the parameter
    :param num_layers: Number of layers
    :param num_chromosomes: Number of chromosomes to generate
    :param logspace: Generate values from logscales
    :param base: Base for log scale space
    :return: An array of array (num_chromosomes, num_layers)
    """
    # Random values
    rand_vals = np.random.uniform(
        low=mini,
        high=maxi,
        size=(num_chromosomes, num_layers)
    )

    # Normal or logspace ?
    if logspace:
        return np.power(base, rand_vals).tolist()
    else:
        return rand_vals.tolist()
    # end if
# end generate_random_parameter_value


# Print population
def print_population(fitness_evaluation):
    """
    Print population
    :param fitness_evaluation: (parameters, fitness_value, model)
    :return:
    """
    for params, fitness_value, _ in fitness_evaluation:
        print("Params : {}, with fitness value : {}".format(params, fitness_value))
    # end for
    print("")
# end print_population


# Exp. parameters
population_size = 20
sample_len = 5000
n_samples_eval = 5
n_samples_model = 5
vocabulary_size = 10
n_layers = 10
n_chromo = 2000
reservoir_size = 10
perturbation_position = 100
fitness_measure = 'TS'
esn_type = 'IF'
selected_population = 5
mutation_prob = 0.1
iterations = 20
num_workers = 6
use_cuda = False and torch.cuda.is_available()
dtype = torch.float64

# Manual seed initialisation
echotorch.utils.manual_seed(1)

# Get a random optimizer
genetic_optimizer = optim.optimizer_factory.get_optimizer(
    'genetic',
    num_workers=num_workers,
    population_size=population_size,
    selected_population=selected_population,
    mutation_probability=mutation_prob,
    iterations=iterations,
    target='max'
)

# Add a hook to see the evaluation
genetic_optimizer.add_hook('evaluation', print_population)

# Create the dataset
random_sequence_dataset = etds.TransformDataset(
    root_dataset=etds.RandomSymbolDataset(
        sample_len=sample_len,
        n_samples=n_samples_model,
        vocabulary_size=10
    ),
    transform=ettr.timeseries.ToOneHot(output_dim=vocabulary_size, dtype=dtype),
    transform_indices=None
)

# Parameters ranges
param_ranges = dict()
param_ranges['w_connectivity'] = generate_random_parameter_value(0.1, 1.0, n_layers, n_chromo)
param_ranges['win_connectivity'] = generate_random_parameter_value(0.1, 1.0, n_layers, n_chromo)
param_ranges['leak_rate'] = generate_random_parameter_value(0.1, 1.0, n_layers, n_chromo)
param_ranges['spectral_radius'] = generate_random_parameter_value(0.01, 2.0, n_layers, n_chromo)
param_ranges['input_scaling'] = generate_random_parameter_value(0.1, 1.0, n_layers, n_chromo)
param_ranges['bias_scaling'] = generate_random_parameter_value(0.0, 1.0, n_layers, n_chromo)

# Launch the optimization of hyperparameters
_, best_param, best_measure = genetic_optimizer.optimize(
    evaluation_function,
    param_ranges,
    random_sequence_dataset,
    n_layers=n_layers,
    n_samples=n_samples_eval,
    reservoir_size=reservoir_size,
    sample_len=sample_len,
    perturbation_position=perturbation_position,
    vocabulary_size=vocabulary_size,
    samples_per_model=n_samples_model,
    fitness_measure=fitness_measure,
    esn_type=esn_type,
    use_cuda=use_cuda
)

# Show the result
print("Best hyper-parameters found : {}".format(best_param))
print("Best {} : {}".format(fitness_measure, best_measure))
