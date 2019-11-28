# -*- coding: utf-8 -*-
#

# Import
from .matrix_generation import MatlabLoader, MatrixFactory, MatrixGenerator, NormalMatrixGenerator, NumpyLoader
from .matrix_generation import UniformMatrixGenerator
from .error_measures import nrmse, nmse, rmse, mse, perplexity, cumperplexity, generalized_squared_cosine
from .utility_functions import align_pattern, compute_correlation_matrix, spectral_radius, deep_spectral_radius, \
    normalize, average_prob, max_average_through_time, compute_singular_values, compute_similarity_matrix, \
    pattern_interpolation
from .visualisation import ESNCellObserver, Observable

__all__ = [
    'align_pattern', 'compute_correlation_matrix', 'nrmse', 'nmse', 'rmse', 'mse', 'perplexity', 'cumperplexity',
    'spectral_radius', 'deep_spectral_radius',
    'normalize', 'average_prob', 'max_average_through_time', 'compute_singular_values', 'generalized_squared_cosine',
    'compute_similarity_matrix', 'pattern_interpolation', 'MatlabLoader', 'MatrixFactory', 'MatrixGenerator',
    'NormalMatrixGenerator', 'NumpyLoader', 'UniformMatrixGenerator', 'ESNCellObserver',
    'Observable'
]
