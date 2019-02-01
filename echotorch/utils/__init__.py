# -*- coding: utf-8 -*-
#

# Imports
from .error_measures import nrmse, nmse, rmse, mse, perplexity, cumperplexity, generalized_squared_cosine
from .utility_functions import spectral_radius, deep_spectral_radius, normalize, average_prob, max_average_through_time, compute_singular_values, compute_similarity_matrix, find_phase_shift
from .visualisation import show_3d_timeseries, show_2d_timeseries, show_1d_timeseries, neurons_activities_1d, neurons_activities_2d, neurons_activities_3d, plot_singular_values, show_similarity_matrix, show_conceptors_similarity_matrix, show_sv_for_increasing_aperture

__all__ = [
    'nrmse', 'nmse', 'rmse', 'mse', 'perplexity', 'cumperplexity', 'spectral_radius', 'deep_spectral_radius',
    'normalize', 'average_prob', 'max_average_through_time', 'show_3d_timeseries', 'show_2d_timeseries',
    'show_1d_timeseries', 'neurons_activities_1d', 'neurons_activities_2d', 'neurons_activities_3d',
    'plot_singular_values', 'compute_singular_values', 'generalized_squared_cosine', 'compute_similarity_matrix',
    'show_similarity_matrix', 'show_conceptors_similarity_matrix', 'show_sv_for_increasing_aperture',
    'find_phase_shift'
]
