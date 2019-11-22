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


# Pattern interpolation
def pattern_interpolation(p, y, interpolation_rate, error_measure=nrmse):
    """
    Pattern interpolation
    :param p:
    :param y:
    :param interpolation_rate:
    :param error_measure:
    :return:
    """
    # Length
    CL = y.size(0)
    PL = p.size(0)

    # Interpolation of generated sample
    interpolated_func = interp1d(np.arange(CL), y[:, 0].numpy(), kind='quadratic')
    interpolated_generated = interpolated_func(np.arange(0, CL - 1.0, 1.0 / interpolation_rate))

    # Interpolation of target sample
    interpolated_func = interp1d(np.arange(PL), p.numpy(), kind='quadratic')
    interpolated_pattern = interpolated_func(np.arange(0, PL - 1.0, 1.0 / interpolation_rate))

    # Length of generated (interpolated)
    L = interpolated_generated.shape[0]

    # Length of original (interpolated)
    M = interpolated_pattern.shape[0]

    # Save norm-2 for each phase shift
    norm_phase_shift = np.zeros(L - M)

    # Phase shift
    for shift in range(L - M):
        # Norm-2 between generated an original
        norm_phase_shift[shift] = lin.norm(interpolated_generated[shift:shift + M] - interpolated_pattern)
    # end for

    # Find minimum distance
    min_norm = int(np.argmax(-norm_phase_shift))

    # Generated signal aligned
    generated_sample_aligned = interpolated_generated[
        np.arange(min_norm, min_norm + PL * interpolation_rate, interpolation_rate)
    ]

    # Original phase
    original_phase = np.ceil(min_norm / interpolation_rate)

    # To Tensor
    generated_sample_aligned = torch.Tensor(generated_sample_aligned)

    # Double ?
    if isinstance(generated_sample_aligned, torch.DoubleTensor):
        generated_sample_aligned = generated_sample_aligned.double()
    # end if

    # Error after alignment
    error_aligned = error_measure(generated_sample_aligned.reshape(1, -1), p.reshape(1, -1))

    return generated_sample_aligned, original_phase, error_aligned
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
