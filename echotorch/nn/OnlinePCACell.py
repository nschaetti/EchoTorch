# -*- coding: utf-8 -*-
#
# File : echotorch/nn/ESN.py
# Description : An Echo State Network module.
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
# Copyright Nils Schaetti <nils.schaetti@unine.ch>

"""
Created on 26 January 2018
@author: Nils Schaetti
"""

# Imports
import torch.sparse
import torch
import torch.nn as nn
from torch.autograd import Variable


# Online PCA cell
# We extract the principal components from the input data incrementally.
class OnlinePCACell(nn.Module):
    """
    Online PCA cell
    We extract the principal components from the input data incrementally.
    Weng J., Zhang Y. and Hwang W.,
    Candid covariance-free incremental principal component analysis,
    IEEE Trans. Pattern Analysis and Machine Intelligence,
    vol. 25, 1034--1040, 2003.
    """

    # Constructor
    def __init__(self, input_dim, output_dim, amn_params=(20, 200, 2000, 3), init_eigen_vectors=None, var_rel=1, numx_rng=None):
        """
        Constructor
        :param input_dim:
        :param output_dim:
        :param amn_params:
        :param init_eigen_vectors:
        :param var_rel:
        :param numx_rng:
        """
        # Super call
        super(OnlinePCACell, self).__init__()

        # Properties
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.amn_params = amn_params
        self._init_v = init_eigen_vectors
        self.var_rel = var_rel
        self._train_iteration = 0
        self._training_type = None

        # (Internal) eigenvectors
        self._v = None
        self.v = None
        self.d = None

        # Total and reduced
        self._var_tot = 1.0
        self._reduced_dims = self.output_dim
    # end __init__

    ###############################################
    # PROPERTIES
    ###############################################

    # Initial eigen vectors
    @property
    def init_eigen_vectors(self):
        """
        Initial eigen vectors
        :return:
        """
        return self._init_v
    # end init_eigen_vectors

    # Set initial eigen vectors
    @init_eigen_vectors.setter
    def init_eigen_vectors(self, init_eigen_vectors=None):
        """
        Set initial eigen vectors
        :param init_eigen_vectors:
        :return:
        """
        self._init_v = init_eigen_vectors

        # Set input dim
        if self._input_dim is None:
            self._input_dim = self._init_v.shape[0]
        else:
            # Check input dim
            assert(
                self.input_dim == self._init_v.shape[0]), \
                Exception(u"Dimension mismatch. init_eigen_vectors shape[0] must be {}, given {}".format(
                    self.input_dim,
                    self._init_v.shape[0]
                )
            )
        # end if

        # Set output dim
        if self._output_dim is None:
            self._output_dim = self._init_v.shape[1]
        else:
            # Check output dim
            assert(
                self.output_dim == self._init_v.shape[1],
                Exception(u"Dimension mismatch, init_eigen_vectors shape[1] must be {}, given {}".format(
                    self.output_dim,
                    self._init_v.shape[1])
                )
            )
        # end if

        # Set V
        if self.v is None:
            self._v = self._init_v.copy()
            self.d = torch.norm(self._v, p=2, dim=0)
            self.v = self._v / self.d
        # end if
    # end init_eigen_vectors

    ###############################################
    # PUBLIC
    ###############################################

    # Get variance explained by PCA
    def get_var_tot(self):
        """
        Get variance explained by PCA
        :return:
        """
        return self._var_tot
    # end get_var_tot

    # Get reducible dimensionality based on the set thresholds
    def get_reduced_dimensionality(self):
        """
        Return reducible dimensionality based on the set thresholds.
        :return:
        """
        return self._reduced_dims
    # end get_reduced_dimensionality

    # Get projection matrix
    def get_projmatrix(self, transposed=1):
        """
        Get projection matrix
        :param transposed:
        :return:
        """
        if transposed:
            return self.v
        # end if
        return self.v.t()
    # end get_projmatrix

    # Get back-projection matrix (reconstruction matrix)
    def get_recmatrix(self, transposed=1):
        """
        Get reconstruction matrix
        :param transposed:
        :return:
        """
        if transposed:
            return self.v.t()
        # end if
        return self.v
    # end get_recmatrix

    # Reset learning
    def reset(self):
        """
        Reset learning
        :return:
        """
        # Training mode again
        self.train(True)
    # end reset

    # Forward
    def forward(self, x, y=None):
        """
        Forward
        :param x: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        # Update components
        self._update_pca(x)

        # Execute
        return self._execute(x)
    # end forward

    ###############################################
    # PRIVATE
    ###############################################

    # Project the input on the first 'n' components
    def _execute(self, x, n=None):
        """
        Project the input on the first 'n' components
        :param x:
        :param n:
        :return:
        """
        if n is not None:
            return x.mm(self.v[:, :n])
        # end if
        return x.mm(self.v)
    # end _execute

    # Update the principal components.
    def _update_pca(self, x):
        """
        Update the principal components
        :param x:
        :return:
        """
        # Params
        [w1, w2] = self._amnesic(self.get_current_train_iteration() + 1)
        red_j = self.output_dim
        red_j_flag = False
        explained_var = 0.0

        # For each output
        r = x
        for j in range(self.output_dim):
            v = self._v[:, j:j + 1]
            d = self.d[j]

            v = w1 * v + w2 * r.mv(v) / d * r.t()
            d = torch.norm(v)
            vn = v / d
            r = r - r.mv(vn) * vn.t()
            explained_var += d

            # Red flag
            if not red_j_flag:
                ratio = explained_var / self._var_tot
                if ratio > self.var_rel:
                    red_j = j
                    red_j_flag = True
                # end if
            # end if

            self._v[:, j:j + 1] = v
            self.v[:, j:j + 1] = vn
            self.d[j] = d
        # end for

        self._var_tot = explained_var
        self._reduced_dims = red_j
    # end update_pca

    # Initialize parameters
    def _check_params(self, *args):
        """
        Initialize parameters
        :param args:
        :return:
        """
        if self._init_v is None:
            if self.output_dim is not None:
                self.init_eigen_vectors = 0.1 * torch.randn(self.input_dim, self.output_dim)
            else:
                self.init_eigen_vectors = 0.1 * torch.randn(self.input_dim, self.input_dim)
            # end if
        # end if
    # end _check_params

    # Return amnesic weights
    def _amnesic(self, n):
        """
        Return amnesic weights
        :param n:
        :return:
        """
        _i = float(n + 1)
        n1, n2, m, c = self.amn_params
        if _i < n1:
            l = 0
        elif (_i >= n1) and (_i < n2):
            l = c * (_i - n1) / (n2 - n1)
        else:
            l = c + (_i - n2) / m
        # end if
        _world = float(_i - 1 - l) / _i
        _wnew = float(1 + l) / _i
        return [_world, _wnew]
    # end _amnesic

    # Add constant
    def _add_constant(self, x):
        """
        Add constant
        :param x:
        :return:
        """
        bias = Variable(torch.ones((x.size()[0], x.size()[1], 1)), requires_grad=False)
        return torch.cat((bias, x), dim=2)
    # end _add_constant

# end PCACell
