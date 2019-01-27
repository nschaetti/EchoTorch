# -*- coding: utf-8 -*-
#

# Imports
import torch
from sklearn.decomposition import PCA
import numpy as np


# Compute singular values
def compute_singular_values(stats, aperture=-1):
    """
    Compute singular values
    :param states:
    :return:
    """
    # PCA
    pca = PCA(n_components=stats.shape[1], svd_solver='full')
    pca.fit(stats)

    # Singular values
    if aperture == -1:
        return pca.singular_values_, pca.components_
    else:
        return pca.singular_values_ / (pca.singular_values_ + (1.0 / np.power(aperture, 2))), pca.components_
    # end if
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
