# -*- coding: utf-8 -*-
#
# File : echotorch/utils/optimization/Optimizer.py
# Description : Optimizer base class
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
import math
import concurrent.futures


# Optimizer base class
class Optimizer(object):
    """
    Optimizer base class
    """

    # Constructor
    def __init__(self, num_workers=1, **kwargs):
        """
        Constructor
        :param kwargs: Parameters for the optimizer
        """
        # Workers
        self._num_workers = num_workers

        # Default generation parameters
        self._parameters = dict()
        self._parameters['target'] = 'min'

        # Initialize hooks
        self._hooks = dict()

        # Set parameter values given
        for key, value in kwargs.items():
            self._parameters[key] = value
        # end for
    # end __init__

    #region PROPERTIES

    # Parameters
    @property
    def parameters(self):
        """
        Parameters
        :return: Optimizer's parameters
        """
        return self._parameters
    # end parameters

    # List of hooks (to override)
    @property
    def hooks_list(self):
        """
        List of hooks
        :return: List of hooks
        """
        return []
    # end hooks_list

    #endregion PROPERTIES

    #region PUBLIC

    # Get a parameter value
    def get_parameter(self, key):
        """
        Get a parameter value
        :param key: Parameter name
        :return: Parameter value
        """
        try:
            return self._parameters[key]
        except KeyError:
            raise Exception("Unknown optimizer parameter : {}".format(key))
        # end try
    # end get_parameter

    # Set a parameter value
    def set_parameter(self, key, value):
        """
        Set a parameter value
        :param key: Parameter name
        :param value: Parameter value
        """
        try:
            self._parameters[key] = value
        except KeyError:
            raise Exception("Unknown optimizer parameter : {}".format(key))
        # end try
    # end set_parameter

    # Optimize
    def optimize(self, test_function, param_ranges, datasets, *args, **kwargs):
        """
        Optimize
        :param test_function: The function that maps a list of parameters, training samples, test samples,
        and their corresponding ground truth to a measured fitness.
        :param param_ranges: A dictionary with parameter names and ranges
        :param datasets: A tuple with dataset used to train and test the model as a list of tuples (X, Y) with X,
        and Y the target to be learned. (training dataset, test dataset) or
        (training dataset, dev dataset, test dataset)
        :return: Three objects, the model object, the best parameter values as a dict,
        the fitness value obtained by the best model.
        """
        return self._optimize_func(test_function, param_ranges, datasets, *args, **kwargs)
    # end optimize

    # Add a hook function
    def add_hook(self, hook_name, hook_func):
        """
        Add a hook function
        :param hook_name: Hook's name
        :param hook_func: Hook's function
        """
        # Check hook exists
        if hook_name in self.hooks_list:
            # Add a list if it does not exists
            if hook_name not in self._hooks:
                self._hooks[hook_name] = list()
            # end if

            # Add the function
            self._hooks[hook_name].append(hook_func)
        else:
            raise Exception("Unknown hook : {}".format(hook_name))
        # end if
    # end add_hook

    #endregion PUBLIC

    #region PRIVATE

    # Call a hook
    def _call_hook(self, hook_name, *args, **kwargs):
        """
        Call a hook
        :param hook_name: Hook's name
        :param args:
        :param kwargs:
        :return:
        """
        # Check hook exists
        if hook_name in self.hooks_list:
            # Check that function are registrered
            if hook_name in self._hooks.keys():
                # Get function
                hook_funcs = self._hooks[hook_name]

                # Call each hook
                for hook_func in hook_funcs:
                    hook_func(*args, **kwargs)
                # end for
            # end if
        else:
            raise Exception("Unknown hook : {}".format(hook_name))
        # end if
    # end _call_hook

    # Evaluate a set of parameters with multiple threads
    def _evaluate_with_workers(self, worker_func, params_set, *args, **kwargs):
        """
        Evaluate a set of parameters with multiple threads
        :param params_set:
        :param worker_func:
        :param args:
        :param kwargs:
        :return:
        """
        # How many pass
        n_passes = int(math.ceil(len(params_set) / self._num_workers))

        # Overall results
        overall_results = list()

        # Position in param set
        param_i = 0

        # For each pass
        for pass_i in range(n_passes):
            # Create an executor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # List of thread
                thread_list = list()

                # For each worker
                for worker_i in range(self._num_workers):
                    # Launch threads
                    thread_list.append(executor.submit(worker_func, params_set[param_i], *args, **kwargs))
                    param_i += 1
                # end for
            # end with

            # Get results
            for thread_i, t in enumerate(thread_list):
                # Results
                model, fitness_measure = t.result()

                # Append
                overall_results.append(
                    (params_set[param_i-self._num_workers+thread_i], fitness_measure, model)
                )
            # end for

        # end for

        return overall_results
    # end _evaluate_with_workers

    # Optimize function to override
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
        pass
    # end _optimize_func

    # Set parameters
    def _set_parameters(self, args):
        """
        Set parameters
        :param args: Parameters as dict
        """
        for key, value in args.items():
            self.set_parameter(key, value)
        # end for
    # end _set_parameters

    #endregion PRIVATE

# end Optimizer
