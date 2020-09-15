# -*- coding: utf-8 -*-
#
# File : echotorch/utils/optimization/GeneticOptimizer.py
# Description : Optimize parameters with a genetic algorithm.
# Date : 18 August, 2020
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
import random
import math
from .Optimizer import Optimizer
from .OptimizerFactory import optimizer_factory


# Hyperparameters optimization based on evolutionary computing
class GeneticOptimizer(Optimizer):
    """
    Hyperparameters optimization based on evolutionary computing
    """

    # Constructor
    def __init__(self, num_workers=1, **kwargs):
        """
        Constructor
        :param kwargs: Argument for the optimizer
        """
        # Set default parameter values
        super(GeneticOptimizer, self).__init__(
            num_workers=num_workers,
            population_size=20,
            selected_population=5,
            mutation_probability=0.1,
            iterations=10
        )

        # Set parameters
        self._set_parameters(args=kwargs)
    # end __init__

    # region PROPERTIES

    # List of hooks (to override)
    @property
    def hooks_list(self):
        """
        List of hooks
        :return: List of hooks
        """
        return ['population', 'evaluation']
    # end hooks_list

    # endregion PROPERTIES

    # region PRIVATE

    # Generate the initial population
    def _generate_random_population(self, param_ranges):
        """
        Generate the initial random population
        :param param_ranges: Parameters values
        :return:
        """
        # Get how many individual in the initial population
        population_size = self.get_parameter('population_size')

        # Population as a list of individuals
        population = list()

        # For each individual
        for r in range(population_size):
            # Individual : a list of parameter values
            individual = dict()

            # For each parameters
            for param_name, param_range in param_ranges.items():
                # Get a random value for this param
                individual[param_name] = param_range[random.randrange(len(param_range))]
            # end for

            # Add to population
            population.append(individual)
        # end for

        return population
    # end _generate_random_population

    # Make a crossover between two individuals
    def _crossover(self, individual1, individual2):
        """
        Make a crossover between two individuals
        :param individual1: First individual
        :param individual2: Second individual
        :return: A new individual made from the crossover
        """
        # How many parameters there is
        params = individual1.keys()
        n_params = len(params)

        # Generate a random position in the DNA
        random_position = random.randint(1, n_params - 1)

        # The new individual as a dict
        new_individual = dict()

        # For each parameter
        for p_i, param in enumerate(params):
            # Switch between individuals
            new_individual[param] = individual1[param] if p_i < random_position else individual2[param]
        # end for

        return new_individual
    # end _crossover

    # Mutate an individual
    def _mutation(self, individual, mutation_prob, parameter_ranges):
        """
        Mutate an individual
        :param individual: The individual as a dict of values
        :param mutation_prob: The probability that a gene will be mutated
        :param parameter_ranges: The range of possible value for each parameters (dict)
        :return: The mutated individual
        """
        # For each parameter
        for param_name, param_range in parameter_ranges.items():
            # Check probability
            if random.random() < mutation_prob:
                individual[param_name] = param_range[random.randrange(len(param_range))]
            # end if
        # end for

        return individual
    # end _mutation

    # Create a new generation by mating the old one
    def _population_mating(self, population, selected_population_size, new_population_size, mutation_prob,
                           parameter_ranges):
        """
        Create a new generation by maiting the old one
        :param population: The old generation
        :param selected_population_size: Size of the select population
        :param new_population_size: The size of the new generation
        :param mutation_prob: Probability of mutation
        :return: The new generation
        """
        # The new generation
        new_population = list()

        # For each new individual to generate
        for p in range(new_population_size):
            # Select to a random individual
            individual1 = population[random.randrange(0, selected_population_size)]

            # Select another random individual
            individual2 = individual1
            while individual2 == individual1:
                individual2 = population[random.randrange(0, selected_population_size)]
            # end while

            # New individual from crossover
            new_individual = self._crossover(individual1, individual2)

            # Apply mutation
            new_individual = self._mutation(new_individual, mutation_prob, parameter_ranges)

            # Make a crossover between these two individuals
            new_population.append(new_individual)
        # end for

        return new_population
    # end _population_mating

    # Optimize hyper-parameters
    def _optimize_func(self, test_function, param_ranges, datasets, *args, **kwargs):
        """
        Optimize function to override
        :param test_function: The function that maps a list of parameters, training samples, test samples,
        and their corresponding ground truth to a measured fitness.
        :param param_ranges: A dictionary with parameter names and ranges
        :param datasets: A tuple with dataset used to train and test the model as a list of tuples (X, Y) with X,
        and Y the target to be learned. (training dataset, test dataset) or
        (training dataset, dev dataset, test dataset)
        :return: Three objects, the model object, the best parameter values as a dict,
        the fitness value obtained by the best model.
        """
        # Parameters
        population_size = self.get_parameter('population_size')
        iterations = self.get_parameter('iterations')
        selected_population_count = self.get_parameter('selected_population')
        mutation_prob = self.get_parameter('mutation_probability')

        # Get the initial random population
        population = self._generate_random_population(param_ranges)

        # Hook
        self._call_hook('population', population)

        # Keep the best individual
        if self.get_parameter('target') == 'min':
            best_individual = (None, math.inf)
        else:
            best_individual = (None, 0)
        # end if

        # For each iteration
        for epoch in range(iterations):
            # Test each member of the population
            fitness_values = self._evaluate_with_workers(
                test_function,
                population,
                datasets,
                *args,
                **kwargs
            )

            # Sort individuals from the best to the worse
            if self.get_parameter('target') == 'min':
                fitness_values = sorted(fitness_values, key=lambda tup: tup[1])
            else:
                fitness_values = sorted(fitness_values, key=lambda tup: tup[1], reverse=True)
            # end if

            # Hook
            self._call_hook('evaluation', fitness_values)

            # Keep if it is the best ever
            winner = fitness_values[0]
            if (self.get_parameter('target') == 'min' and winner[1] < best_individual[1]) or \
                    (self.get_parameter('target') == 'max' and winner[1] > best_individual[1]):
                best_individual = winner
            # end if

            # Keep only the best
            selected_population = [individual[0] for individual in fitness_values[:selected_population_count]]

            # Make a new generation from this selected population
            new_population = self._population_mating(
                selected_population,
                selected_population_count,
                population_size,
                mutation_prob,
                param_ranges
            )

            # Keep the winer
            if best_individual[0] is not None:
                new_population[-1] = best_individual[0]
            # end if

            # The new generation is now the current one
            population = new_population

            # Hook
            self._call_hook('population', population)
        # end for

        # Get the best model
        model, fitness_value = test_function(best_individual[0], datasets, **kwargs)

        return model, best_individual[0], best_individual[1]
    # end _optimize_func

    # endregion PRIVATE

# end GeneticOptimizer


# Add
optimizer_factory.register_optimizer("genetic", GeneticOptimizer)

