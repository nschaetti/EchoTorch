# -*- coding: utf-8 -*-
#
# File : echotorch/utils/optimization/RandomOptimizer.py
# Description : Hyperparameters optimization based on random generated values.
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
import numpy as np
from .Optimizer import Optimizer
from .OptimizerFactory import optimizer_factory


# Optimize hyper-parameters based on randomly generated values
class RandomOptimizer(Optimizer):
    """
    Optimize hyper-parameters based on randomly generated values
    """

    # Constructor
    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        # Set default parameter values
        super(RandomOptimizer, self).__init__(
            R=10
        )

        # Set parameters
        self._set_parameters(args=kwargs)
    # end __init__

    #region PRIVATE

    # Optimize hyper-parameters
    def _optimize_func(self, test_function, param_ranges, dataset):
        """
        Optimize function to override
        :param test_function: The function that maps a list of parameters, training samples, test samples,
        and their corresponding ground truth to a measured fitness.
        :param param_ranges: A dictionary with parameter names and ranges
        :param dataset: Dataset used to train and test the model as a list of tuples (X, Y) with X
        and Y the target to be learned.
        :return: Three objects, the model object, the best parameter values as a dict,
        the fitness value obtained by the best model.
        """
        # Population of parameter values
        parameter_population = list()

        # Get how many individual we must test
        R = self.get_parameter('R')

        # For each individual
        for r in range(R):
            # Individual : a list of parameter values
            individual = dict()

            # For each parameters
            for param_name, param_range in param_ranges.items():
                # Get a random value for this param
                individual[param_name] = param_range[random.randrange(len(param_range))]
            # end for

            # Add to population
            parameter_population.append(individual)
        # end for

        # Save fitness values
        fitness_values = np.zeros(R)

        # Test each member of the population
        for r, param_individual in enumerate(parameter_population):
            # Test the model
            _, fitness_value = test_function(param_individual, dataset)

            # Save fitness value
            fitness_values[r] = fitness_value
        # end for

        # Get the best parameter values
        if self.get_parameter('target') == 'min':
            best_param = parameter_population[np.argmin(fitness_values)]
        elif self.get_parameter('target') == 'max':
            best_param = parameter_population[np.argmax(fitness_values)]
        else:
            raise Exception("Unknown target value to optimize : {}".format(self.get_parameter('target')))
        # end if

        # Get the best model
    # end _optimize_func

    #endregion PRIVATE

# end RandomOptimizer


# Add
optimizer_factory.register_optimizer("random", RandomOptimizer)
