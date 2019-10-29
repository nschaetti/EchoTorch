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


# Filter the input data through the most significatives principal components.
class PCACell(nn.Module):
    """
    Filter the input data through the most significatives principal components
    """

    # Constructor
    def __init__(self, input_dim, output_dim, svd=False, reduce=False, var_rel=1E-12, var_abs=1E-15, var_part=None):
        """
        Constructor
        :param input_dim:
        :param output_dim:
        :param svd: If True use Singular Value Decomposition instead of the standard eigenvalue problem solver. Use it when PCANode complains about singular covariance matrices.
        :param reduce: Keep only those principal components which have a variance larger than 'var_abs'
        :param val_rel: Variance relative to first principal component threshold. Default is 1E-12.
        :param var_abs: Absolute variance threshold. Default is 1E-15.
        :param var_part: Variance relative to total variance threshold. Default is None.
        """
        # Super
        super(PCACell, self).__init__()

        # Properties
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.svd = svd
        self.var_abs = var_abs
        self.var_rel = var_rel
        self.var_part = var_part
        self.reduce = reduce

        # Set it as buffer
        self.register_buffer('xTx', Variable(torch.zeros(input_dim, input_dim), requires_grad=False))
        self.register_buffer('xTx_avg', Variable(torch.zeros(input_dim), requires_grad=False))

        # Eigen values
        self.d = None

        # Eigen vectors, first index for coordinates
        self.v = None

        # Total variance
        self.total_variance = None

        # Len, average and explained variance
        self.tlen = 0
        self.avg = None
        self.explained_variance = None
    # end __init__

    ###############################################
    # PROPERTIES
    ###############################################

    ###############################################
    # PUBLIC
    ###############################################

    # Reset learning
    def reset(self):
        """
        Reset learning
        :return:
        """
        # Initialize the covariance matrix one for
        # the input data.
        self._init_internals()

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
        # Number of batches
        n_batches = int(x.size()[0])

        # Time length
        time_length = x.size()[1]

        # Outputs
        outputs = Variable(torch.zeros(n_batches, time_length, self.output_dim))
        outputs = outputs.cuda() if x.is_cuda else outputs

        # For each batch
        for b in range(n_batches):
            # Sample
            s = x[b]

            # Train or execute
            if self.training:
                self._update_cov_matrix(s)
            else:
                outputs[b] = self._execute_pca(s)
            # end if
        # end for

        return outputs
    # end forward

    # Finish training
    def finalize(self):
        """
        Finalize training with LU factorization or Pseudo-inverse
        """
        # Reshape average
        xTx, avg, tlen = self._fix(self.xTx, self.xTx_avg, self.tlen)

        # Reshape
        self.avg = avg.unsqueeze(0)

        # We need more observations than variables
        if self.tlen < self.input_dim:
            raise Exception(u"The number of observations ({}) is larger than  the number of input variables ({})".format(self.tlen, self.input_dim))
        # end if

        # Total variance
        total_var = torch.diag(xTx).sum()

        # Compute and sort eigenvalues
        d, v = torch.symeig(xTx, eigenvectors=True)

        # Check for negative eigenvalues
        if float(d.min()) < 0:
            # raise Exception(u"Got negative eigenvalues ({}). You may either set output_dim to be smaller".format(d))
            pass
        # end if

        # Indexes
        indexes = range(d.size(0)-1, -1, -1)

        # Sort by descending order
        d = torch.take(d, Variable(torch.LongTensor(indexes)))
        v = v[:, indexes]

        # Explained covariance
        self.explained_variance = torch.sum(d) / total_var

        # Store eigenvalues
        self.d = d[:self.output_dim]

        # Store eigenvectors
        self.v = v[:, :self.output_dim]

        # Total variance
        self.total_variance = total_var

        # Stop training
        self.train(False)
    # end finalize

    # Get explained variance
    def get_explained_variance(self):
        """
        The explained variance is the fraction of the original variance that can be explained by the
        principal components.
        :return:
        """
        return self.explained_variance
    # end get_explained_variance

    # Get the projection matrix
    def get_proj_matrix(self, tranposed=True):
        """
        Get the projection matrix
        :param tranposed:
        :return:
        """
        # Stop training
        self.train(False)

        # Transposed
        if tranposed:
            return self.v
        # end if
        return self.v.t()
    # end get_proj_matrix

    # Get the reconstruction matrix
    def get_rec_matrix(self, tranposed=1):
        """
        Returns the reconstruction matrix
        :param tranposed:
        :return:
        """
        # Stop training
        self.train(False)

        # Transposed
        if tranposed:
            return self.v.t()
        # end if
        return self.v
    # end get_rec_matrix

    ###############################################
    # PRIVATE
    ###############################################

    # Project the input on the first 'n' principal components
    def _execute_pca(self, x, n=None):
        """
        Project the input on the first 'n' principal components
        :param x:
        :param n:
        :return:
        """
        if n is not None:
            return (x - self.avg).mm(self.v[:, :n])
        # end if
        return (x - self.avg).mm(self.v)
    # end _execute

    # Project data from the output to the input space using the first 'n' components.
    def _inverse(self, y, n=None):
        """
        Project data from the output to the input space using the first 'n' components.
        :param y:
        :param n:
        :return:
        """
        if n is None:
            n = y.shape[1]
        # end if

        if n > self.output_dim:
            raise Exception(u"y has dimension {} but should but at most {}".format(n, self.output_dim))
        # end if

        # Get reconstruction matrix
        v = self.get_rec_matrix()

        # Reconstruct
        if n is not None:
            return y.mm(v[:n, :]) + self.avg
        else:
            return y.mm(v) + self.avg
        # end if
    # end _inverse

    # Adjust output dim
    def _adjust_output_dim(self):
        """
        If the output dimensions is small than the input dimension
        :return:
        """
        # If the number of PC is not specified, keep all
        if self.desired_variance is None and self.ouput_dim is None:
            self.output_dim = self.input_dim
            return None
        # end if

        # Define the range of eigenvalues to compute if the number of PC to keep
        # has been specified directly.
        if self.output_dim is not None and self.output_dim >= 1:
            return (self.input_dim - self.output_dim + 1, self.input_dim)
        else:
            return None
        # end if
    # end _adjust_output_dim

    # Fix covariance matrix
    def _fix(self, mtx, avg, tlen, center=True):
        """
        Returns a triple containing the covariance matrix, the average and
        the number of observations.
        :param mtx:
        :param center:
        :return:
        """
        mtx /= tlen - 1

        # Substract the mean
        if center:
            avg_mtx = torch.ger(avg, avg)
            avg_mtx /= tlen * (tlen - 1)
            mtx -= avg_mtx
        # end if

        # Fix the average
        avg /= tlen

        return mtx, avg, tlen
    # end fix

    # Update covariance matrix
    def _update_cov_matrix(self, x):
        """
        Update covariance matrix
        :param x:
        :return:
        """
        # Init
        if self.xTx is None:
            self._init_internals()
        # end if

        # Update
        self.xTx.data.add_(x.t().mm(x).data)
        self.xTx_avg.add_(torch.sum(x, dim=0))
        self.tlen += x.size(0)
    # end _update_cov_matrix

    # Initialize covariance
    def _init_cov_matrix(self):
        """
        Initialize covariance matrix
        :return:
        """
        self.xTx.data = torch.zeros(self.input_dim, self.input_dim)
        self.xTx_avg.data = torch.zeros(self.input_dim)
    # end _init_cov_matrix

    # Initialize internals
    def _init_internals(self):
        """
        Initialize internals
        :param x:
        :return:
        """
        # Init covariance matrix
        self._init_cov_matrix()
    # end _init_internals

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
