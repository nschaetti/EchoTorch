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
from .training_and_evaluation import fit, eval, cross_val_score


# All echotorch's modules
__all__ = ['datasets', 'evaluation', 'models', 'nn', 'transforms', 'utils', 'fit', 'eval', 'cross_val_score']
