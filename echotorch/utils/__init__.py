# -*- coding: utf-8 -*-
#
# File : utils/__init__.py
# Description : Utils subpackage init file
# Date : 27th of April, 2021
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

# Matrix generation
# from .matrix_generation import MatlabLoader, MatrixFactory, MatrixGenerator, NormalMatrixGenerator, NumpyLoader
# from .matrix_generation import UniformMatrixGenerator

# Error measure
from .error_measures import nrmse, nmse, rmse, mse, perplexity, cumperplexity, generalized_squared_cosine

# Random functions
from .random import manual_seed

# Utility function
from .utility_functions import align_pattern, compute_correlation_matrix, spectral_radius, deep_spectral_radius, \
    normalize, average_prob, max_average_through_time, compute_singular_values, compute_similarity_matrix, \
    pattern_interpolation, find_pattern_interpolation, find_pattern_interpolation_threshold, quota, rank, \
    entropy

# ALL
__all__ = [
    # Error measures
    'nrmse', 'nmse', 'rmse', 'mse', 'perplexity', 'cumperplexity', 'generalized_squared_cosine',
    # Random functions
    'manual_seed',
    # Utility functions
    'align_pattern', 'compute_correlation_matrix', 'spectral_radius', 'deep_spectral_radius', 'normalize',
    'average_prob', 'max_average_through_time', 'compute_singular_values', 'compute_similarity_matrix',
    'pattern_interpolation', 'find_pattern_interpolation', 'find_pattern_interpolation_threshold', 'quota', 'rank',
    'entropy'
]
