# -*- coding: utf-8 -*-
#
# File : echotorch/utils/optimization/GridSearchOptimizer.py
# Description : Hyperparameters optimization based exhaustive search
# Date : 20 August, 2020
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
import numpy as np
import math
from itertools import product
from .Optimizer import Optimizer
from .OptimizerFactory import optimizer_factory


# Hyperparameters optimization based exhaustive search
class GridSearchOptimizer(Optimizer):
    """
    Hyperparameters optimization based exhaustive search
    """

    # Constructor
    def __init__(self, num_workers=1, **kwargs):
        """
        Constructor
        :param kwargs: Argument for the optimizer
        """
        # Set default parameter values
        super(GridSearchOptimizer, self).__init__(num_workers=num_workers)

        # Set parameters
        self._set_parameters(args=kwargs)

    # end __init__

    # region PRIVATE

    # Convert parameter range to list
    def _convert_parameter_range(self, param_ranges):
        """
        Convert parameter range to list
        :param param_ranges: Dictionary with hyperparameters values
        :return: A dictionary of list, the number of possible combination
        """
        # Converted dictionary
        output_dictionary = {}

        # We also return how many combination there is
        comb_count = 1.0

        # For each item
        for key, value in param_ranges.items():
            if type(value) is list:
                output_dictionary[key] = value
            elif type(value) is int:
                output_dictionary[key] = [value]
            elif type(value) is float:
                output_dictionary[key] = [value]
            elif type(value) is np.ndarray or type(value) is np.array:
                output_dictionary[key] = value.tolist()
            elif type(value) is torch.tensor:
                output_dictionary[key] = value.tolist()
            # end if

            # Multiply the counter of combination
            comb_count *= len(output_dictionary[key])
        # end for

        return output_dictionary, int(comb_count)

    # end _convert_parameter_range

    # Optimize hyper-parameters
    def _optimize_func(self, test_function, param_ranges, datasets, **kwargs):
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
        # Convert to a dictionary of lists
        param_ranges_list, comb_count = self._convert_parameter_range(param_ranges)

        # Population of parameter values
        parameter_population = (
            dict(zip(param_ranges_list.keys(), values)) for values in product(*param_ranges_list.values())
        )

        # Save fitness values
        winner = (None, math.inf)

        # Test each member of the population
        for r, param_individual in enumerate(parameter_population):
            # Test the model
            _, fitness_value = test_function(param_individual, datasets, **kwargs)

            # Keep if it is the best
            if (self.get_parameter('target') == 'min' and fitness_value < winner[1]) or \
                    (self.get_parameter('target') == 'max' and fitness_value > winner[1]):
                winner = (param_individual, fitness_value)
            # end if
        # end for

        # Get the best model
        model, fitness_value = test_function(winner[0], datasets, **kwargs)

        return model, winner[0], fitness_value
    # end _optimize_func

    # endregion PRIVATE

# end GridSearchOptimizer


# Add
optimizer_factory.register_optimizer("grid-search", GridSearchOptimizer)
