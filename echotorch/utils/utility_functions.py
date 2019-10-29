# -*- coding: utf-8 -*-
#

# Imports
import torch
import numpy as np
from .error_measures import nrmse, generalized_squared_cosine
from scipy.interpolate import interp1d
import numpy.linalg as lin


# Compute correlation matrix
def compute_correlation_matrix(states):
    """
    Compute correlation matrix
    :param states:
    :return:
    """
    return states.t().mm(states) / float(states.size(0))
# end compute_correlation_matrix


# Align pattern
def align_pattern(interpolation_rate, truth_pattern, generated_pattern):
    """
    Align pattern
    :param interpolation_rate:
    :param truth_pattern:
    :param generated_pattern:
    :return:
    """
    # Length
    truth_length = truth_pattern.size(0)
    generated_length = generated_pattern.size(0)

    # Remove useless dimension
    truth_pattern = truth_pattern.view(-1)
    generated_pattern = generated_pattern.view(-1)

    # Quadratic interpolation functions
    truth_pattern_func = interp1d(np.arange(truth_length), truth_pattern.numpy(), kind='quadratic')
    generated_pattern_func = interp1d(np.arange(generated_length), generated_pattern.numpy(), kind='quadratic')

    # Get interpolated patterns
    truth_pattern_int = truth_pattern_func(np.arange(0, truth_length - 1.0, 1.0 / interpolation_rate))
    generated_pattern_int = generated_pattern_func(np.arange(0, generated_length - 1.0, 1.0 / interpolation_rate))

    # Generated interpolated pattern length
    L = generated_pattern_int.shape[0]

    # Truth interpolated pattern length
    M = truth_pattern_int.shape[0]

    # Save L2 distance for each phase shift
    phase_matches = np.zeros(L - M)

    # For each phase shift
    for phases_hift in range(L - M):
        phase_matches[phases_hift] = lin.norm(truth_pattern_int - generated_pattern_int[phases_hift:phases_hift + M])
    # end for

    # Best match
    max_ind = int(np.argmax(-phase_matches))

    # Get the position in the original signal
    coarse_max_ind = int(np.ceil(max_ind / interpolation_rate))

    # Get the generated output matching the original signal
    generated_aligned = generated_pattern_int[
        np.arange(max_ind, max_ind + interpolation_rate * truth_length, interpolation_rate)
    ]

    return max_ind, coarse_max_ind, torch.from_numpy(generated_aligned).view(-1, 1)
# end align_pattern


# Find phase shift
def find_phase_shift(p, y, interpolation_rate, error_measure=nrmse):
    """
    Find phase shift
    :param s1:
    :param s2:
    :param window_size:
    :return:
    """
    # Size
    p_length = p.size(0)
    y_length = y.size(0)

    # 1D
    p = p.view(-1)
    y = y.view(-1)

    # Interpolate p and y
    p_int = torch.from_numpy(np.interp(np.arange(0, p_length, 1.0 / interpolation_rate), np.arange(p_length), p.numpy()))
    y_int = torch.from_numpy(np.interp(np.arange(0, y_length, 1.0 / interpolation_rate), np.arange(y_length), y.numpy()))

    # New shape
    L = y_int.shape[0]
    M = p_int.shape[0]

    # Find best phase
    phasematches = torch.zeros(L - M)
    for phaseshift in range(L - M):
        phasematches[phaseshift] = torch.norm(p_int - y_int[phaseshift:phaseshift + M], p=2)
    # end for

    # Best phase
    max_index = torch.argmax(-phasematches)
    # Matching phase
    y_aligned = y_int[np.arange(max_index, max_index + interpolation_rate * p_length, interpolation_rate)]

    # Original phase
    original_phase = np.ceil(max_index / interpolation_rate)

    # Error after alignment
    error_aligned = error_measure(y_aligned.reshape(1, -1), p.reshape(1, -1))

    return p, y_aligned, original_phase, error_aligned
# end find_phase_shift


# Compute similarity matrix
def compute_similarity_matrix(svd_list):
    """
    Compute similarity matrix
    :param svd_list:
    :return:
    """
    # N samples
    n_samples = len(svd_list)

    # Similarity matrix
    sim_matrix = torch.zeros(n_samples, n_samples)

    # For each combinasion
    for i, (Sa, Ua) in enumerate(svd_list):
        for j, (Sb, Ub) in enumerate(svd_list):
            sim_matrix[i, j] = generalized_squared_cosine(Sa, Ua, Sb, Ub)
        # end for
    # end for

    return sim_matrix
# end compute_similarity_matrix


# Compute singular values
def compute_singular_values(stats):
    """
    Compute singular values
    :param states:
    :return:
    """
    # Compute R (correlation matrix)
    R = stats.t().mm(stats) / stats.shape[0]

    # Compute singular values
    return torch.svd(R)
# end compute_singular_values


# Compute spectral radius of a square 2-D tensor
def spectral_radius(m):
    """
    Compute spectral radius of a square 2-D tensor
    :param m: squared 2D tensor
    :return:
    """
    return torch.max(torch.abs(torch.eig(m)[0])).item()
# end spectral_radius


# Compute spectral radius of a square 2-D tensor for stacked-ESN
def deep_spectral_radius(m, leaky_rate):
    """
    Compute spectral radius of a square 2-D tensor for stacked-ESN
    :param m: squared 2D tensor
    :param leaky_rate: Layer's leaky rate
    :return:
    """
    return spectral_radius((1.0 - leaky_rate) * torch.eye(m.size(0), m.size(0)) + leaky_rate * m)
# end spectral_radius


# Normalize a tensor on a single dimension
def normalize(tensor, dim=1):
    """
    Normalize a tensor on a single dimension
    :param t:
    :return:
    """
    pass
# end normalize


# Average probabilties through time
def average_prob(tensor, dim=0):
    """
    Average probabilities through time
    :param tensor:
    :param dim:
    :return:
    """
    return torch.mean(tensor, dim=dim)
# end average_prob


# Max average through time
def max_average_through_time(tensor, dim=0):
    """
    Max average through time
    :param tensor:
    :param dim: Time dimension
    :return:
    """
    average = torch.mean(tensor, dim=dim)
    return torch.max(average, dim=dim)[1]
# end max_average_through_time
