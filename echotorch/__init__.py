# -*- coding: utf-8 -*-
#
# File : echotorch/__init__.py
# Description : EchoTorch package init file.
# Date : 26th of January, 2018
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
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>

# Imports
from . import datasets
from . import evaluation
from . import models
from . import nn
from . import transforms
from . import utils

# Utility functions
from .conceptors import conceptor, cone, czero, cidentity, OR, AND, NOT, PHI, conceptor_set
from .conceptors import csim, csimilarity
from .matrices import cycle_with_jumps, cycle_with_jumps_generator, matlab, matlab_generator, normal, normal_generator
from .matrices import uniform, uniform_generator
from .modules import esn
from .series import copytask, cross_eval, delaytask, discrete_markov_chain, csv_file, henon, segment_series
from .tensor import TimeTensor
from .tensor_utils import from_numpy
from .training_and_evaluation import fit, eval, cross_val_score
from .utility_functions import timetensor, timecat


# All echotorch's modules
__all__ = [
    'esn', 'TimeTensor', 'timetensor', 'datasets', 'evaluation', 'models', 'nn', 'transforms', 'utils', 'fit', 'eval',
    'cross_val_score', 'timecat', 'copytask', 'discrete_markov_chain', 'csv_file', 'henon', 'from_numpy',
    'delaytask', 'cross_eval', 'segment_series', 'cycle_with_jumps', 'matlab', 'normal', 'uniform',
    'cycle_with_jumps_generator', 'matlab_generator', 'normal_generator', 'uniform_generator', 'conceptor', 'cone',
    'czero', 'cidentity', 'OR', 'AND', 'NOT', 'PHI', 'conceptor_set', 'csim', 'csimilarity'
]
