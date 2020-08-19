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


# Optimizer base class
class Optimizer(object):
    """
    Optimizer base class
    """

    # Constructor
    def __init__(self, **kwargs):
        """
        Constructor
        :param kwargs: Parameters for the optimizer
        """
        # Default generation parameters
        self._parameters = dict()
        self._parameters['target'] = 'min'

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
    def optimize(self, test_function, param_ranges, samples):
        """
        Optimize
        :param test_function: The function that maps a list of parameters, training samples, test samples,
        and their corresponding groundtruth to a measured fitness.
        :param param_ranges: A dictionnary with parameter names and ranges
        :param samples: Data used to train and test the model as a list of tuples (X, Y) with X
        and Y the target to be learned.
        :return: Three objects, the model object, the best parameter values as a dict,
        the fitness value obtained by the best model.
        """
        self._optimize_func(test_function, param_ranges, samples)
    # end optimize

    #endregion PUBLIC

    #region PRIVATE

    # Optimize function to override
    def _optimize_func(self, test_function, param_ranges, samples):
        """
        Optimize function to override
        :param test_function: The function that maps a list of parameters, training samples, test samples,
        and their corresponding groundtruth to a measured fitness.
        :param param_ranges: A dictionnary with parameter names and ranges
        :param samples: Data used to train and test the model as a list of tuples (X, Y) with X
        and Y the target to be learned.
        :return: Three objects, the model object, the best parameter values as a dict,
        the fitness value obtained by the best model.
        """
        pass
    # end _optimize_func

    #endregion PRIVATE

# end Optimizer
