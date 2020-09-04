# -*- coding: utf-8 -*-
#
# File : echotorch/utils/optimization/__init__.py
# Description : Package for optimization of hyperparameters.
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
from .Optimizer import Optimizer
from .OptimizerFactory import OptimizerFactory, optimizer_factory
from .GeneticOptimizer import GeneticOptimizer
from .GridSearchOptimizer import GridSearchOptimizer
from .RandomOptimizer import RandomOptimizer

# ALL
__all__ = ['Optimizer', 'OptimizerFactory', 'GeneticOptimizer', 'GridSearchOptimizer', 'RandomOptimizer']
