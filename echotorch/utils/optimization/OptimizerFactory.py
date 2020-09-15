# -*- coding: utf-8 -*-
#
# File : echotorch/utils/optimization/OptimizerFactory.py
# Description : Class to instanciate optimizers
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


# Optimizer factory
class OptimizerFactory(object):
    """
    Optimizer factory
    """

    # Instance
    _instance = None

    # Constructor
    def __init__(self):
        """
        Constructor
        """
        # Init. optimizers
        self._optimizers = {}

        # Save instance
        self._instance = self
    # end __init__

    #region PUBLIC

    # Register optimizer
    def register_optimizer(self, name, optimizer):
        """
        Register optimizer
        :param name: Optimizer's name
        :param optimizer: Optimizer object
        """
        self._optimizers[name] = optimizer
    # end register_optimizer

    # Get an optimizer
    def get_optimizer(self, name, *args, num_workers=1, **kwargs):
        """
        Get an optimizer
        :param name: Optimizer's name
        :param num_workers: How many thread to use for evaluation
        :param kwargs: Arguments for the optimizer
        """
        optimizer = self._optimizers[name]
        if not optimizer:
            raise ValueError(name)
        # end if
        return optimizer(num_workers, *args, **kwargs)
    # end get_optimizer

    #endregion PUBLIC

    #region STATIC

    # Get instance
    def get_instance(self):
        """
        Get factory's instance
        :return: Instance
        """
        return self._instance
    # end get_instance

    #endregion STATIC

# end OptimizerFactory


# Create the factory
optimizer_factory = OptimizerFactory()
