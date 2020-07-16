# -*- coding: utf-8 -*-
#
# File : echotorch/utils/matrix_generation/__init__.py
# Description : Matrix generation subpackage init file.
# Date : 29th of October, 2019
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
# Copyright Nils Schaetti <nils.schaetti@unine.ch>

# Import basis
from .AperiodicSequenceMatrixGenerator import AperiodicSequenceMatrixGenerator
from .CycleWithJumpsMatrixGenerator import CycleWithJumpsMatrixGenerator
from .MatrixFactory import MatrixFactory, matrix_factory
from .MatrixGenerator import MatrixGenerator
from .MatloabLoader import MatlabLoader
from .NormalMatrixGenerator import NormalMatrixGenerator
from .NumpyLoader import NumpyLoader
from .UniformMatrixGenerator import UniformMatrixGenerator

# All
__all__ = [
    'AperiodicSequenceMatrixGenerator', 'CycleWithJumpsMatrixGenerator', 'MatrixFactory', 'MatrixGenerator',
    'MatloabLoader', 'NumpyLoader', 'matrix_factory'
]
