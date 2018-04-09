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
import numpy as np
from past.utils import old_div


# Slow Feature Analysis layer
class SFACell(nn.Module):
    """
    Extract the slowly varying components from input data.
    """

    # Type keys
    _type_keys = ['f', 'd', 'F', 'D']

    # Type conv
    _type_conv = {('f', 'd'): 'd', ('f', 'F'): 'F', ('f', 'D'): 'D',
                  ('d', 'F'): 'D', ('d', 'D'): 'D',
                  ('F', 'd'): 'D', ('F', 'D'): 'D'}

    # Constructor
    def __init__(self, input_dim, output_dim, include_last_sample=True, rank_deficit_method='none', use_bias=True):
        """
        Constructor
        :param input_dim: Input dimension
        :param output_dim: Number of slow feature
        :param include_last_sample: If set to False, the training method discards the last sample in every chunk during training when calculating the matrix.
        :param rank_deficit_method: 'none', 'reg', 'pca', 'svd', 'auto'.
        """
        super(SFACell, self).__init__()
        self.include_last_sample = include_last_sample
        self.use_bias = use_bias
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialie the two covariance matrices one for
        # the input data, and the other for the derivatives.
        self.xTx = torch.zeros(input_dim, input_dim)
        self.xTx_avg = torch.zeros(input_dim)
        self.dxTdx = torch.zeros(input_dim, input_dim)
        self.dxTdx_avg = torch.zeros(input_dim)

        # Set routine for eigenproblem
        self.set_rank_deficit_method(rank_deficit_method)
        self.rank_threshold = 1e-12
        self.rank_deficit = 0

        # Will be set after training
        self.d = None
        self.sf = None
        self.avg = None
        self.bias = None
        self.tlen = None
    # end __init__

    ###############################################
    # PROPERTIES
    ###############################################

    ###############################################
    # PUBLIC
    ###############################################

    # Time derivative
    def time_derivative(self, x):
        """
        Compute the approximation of time derivative
        :param x:
        :return:
        """
        return x[1:, :] - x[:-1, :]
    # end time_derivative

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
    def forward(self, x):
        """
        Forward
        :param x: Input signal.
        :return: Output or hidden states
        """
        # For each batch
        for b in np.arange(0, x.size(0)):
            # If training or execution
            if self.training:
                # Last sample
                last_sample_index = None if self.include_last_sample else -1

                # Sample and derivative
                xs = x[b, :last_sample_index, :]
                xd = self.time_derivative(x[b])

                # Update covariance matrix
                self.xTx.data.add(xs.t().mm(xs))
                self.dxTdx.data.add(xd.t().mm(xd))

                # Update average
                self.xTx_avg += torch.sum(xs, axis=1)
                self.dxTdx_avg += torch.sum(xd, axis=1)

                # Length
                self.tlen += x.size(0)
            else:
                x[b].mv(self.sf) - self.bias
            # end if
        # end if
        return x
    # end forward

    # Finish training
    def finalize(self):
        """
        Finalize training with LU factorization or Pseudo-inverse
        """
        # Covariance
        self.xTx, self.xTx_avg, self.tlen = self._fix(self.xtX, self.xTx_avg, self.tlen, center=True)
        self.dxTdx, self.dxTdx_avg, self.tlen = self._fix(self.dxTdx, self.dxTdx_avg, self.tlen, center=False)

        # Range
        rng = (1, self.output_dim)

        # Resolve system
        self.d, self.sf = self._symeig(
            self.dxTdx, self.xTx, rng
        )
        d = self.d

        # We want only positive values
        if torch.min(d) < 0:
            raise Exception(u"Got negative values in {}".format(d))
        # end if

        # Delete covariance matrix
        del self.xTx
        del self.dxTdx

        # Store bias
        self.bias = self.xTx_avg * self.sf
    # end finalize

    ###############################################
    # PRIVATE
    ###############################################

    # Solve standard and generalized eigenvalue problem for symmetric (hermitian) definite positive matrices
    def _symeig(self, A, B, range, eigenvectors=True):
        """
        Solve standard and generalized eigenvalue problem for symmetric (hermitian) definite positive matrices.
        :param A: An N x N matrix
        :param B: An N x N matrix
        :param range: (lo, hi), the indexes of smallest and largest eigenvalues to be returned.
        :param eigenvectors: Return eigenvalues and eigenvector or only engeivalues
        :return: w, the eigenvalues and Z the eigenvectors
        """
        # To numpy
        A = A.numpy()
        B = B.numpy()

        # Type
        dtype = np.dtype()

        # Make B the identity matrix
        wB, ZB = np.linalg.eigh(B)

        # Check eigenvalues
        self._assert_eigenvalues_real(wB)

        # No negative values
        if wB.real.min() < 0:
            raise Exception(u"Got negative eigenvalues: {}".format(wB))
        # end if

        # Old division
        ZB = old_div(ZB.real, np.sqrt(wB.real))

        # A = ZB^T * A * ZB
        A = np.matmul(np.matmul(ZB.T, A), ZB)

        # Diagonalize A
        w, ZA = np.linalg.eigh(A)
        Z = np.matmul(ZB, ZA)

        # Check eigenvalues
        self._assert_eigenvalues_real(w, dtype)

        # Read
        w = w.real
        Z = Z.real

        # Sort
        idx = w.argsort()
        w = w.take(idx)
        Z = Z.take(idx, axis=1)

        # Sanitize range
        n = A.shape[0]
        lo, hi = range
        if lo < 1:
            lo = 1
        # end if
        if lo > n:
            lo = n
        # end if
        if hi > n:
            hi = n
        # end if
        if lo > hi:
            lo, hi = hi, lo
        # end if

        # Get values
        Z = Z[:, lo-1:hi]
        w = w[lo-1:hi]

        # Cast
        w = self.refcast(w, dtype)
        Z = self.refcast(Z, dtype)

        # Eigenvectors
        if eigenvectors:
            return torch.FloatTensor(w), torch.FloatTensor(Z)
        else:
            return torch.FloatTensor(w)
        # end if
    # end _symeig

    # Ref cast
    def refcast(self, array, dtype):
        """
        Cast the array to dtype only if necessary, otherwise return a reference.
        """
        dtype = np.dtype(dtype)
        if array.dtype == dtype:
            return array
        return array.astype(dtype)
    # end refcast

    # Check eigenvalues
    def _assert_eigenvalues_real(self, w, dtype):
        """
        Check eigenvalues
        :param w:
        :param dtype:
        :return:
        """
        tol = np.finfo(dtype.type).eps * 100
        if abs(w.imag).max() > tol:
            err = "Some eigenvalues have significant imaginary part: %s " % str(w)
            raise Exception(err)
        # end if
    # end _assert_eigenvalues_real

    # Greatest common type
    def _greatest_common_dtype(self, alist):
        """
        Apply conversion rules to find the common conversion type
        dtype 'd' is default for 'i' or unknown types
        (known types: 'f','d','F','D').
        """
        dtype = 'f'
        for array in alist:
            if array is None:
                continue
            tc = array.dtype.char
            if tc not in self._type_keys:
                tc = 'd'
            transition = (dtype, tc)
            if transition in self._type_conv:
                dtype = self._type_conv[transition]
        return dtype
    # end _greatest_common_dtype

    # Fix covariance matrix
    def _fix(self, mtx, avg, tlen, center=True):
        """
        Returns a triple containing the covariance matrix, the average and
        the number of observations.
        :param mtx:
        :param center:
        :return:
        """
        if self.use_bias:
            mtx /= tlen
        else:
            mtx /= tlen - 1
        # end if

        # Substract the mean
        if center:
            avg_mtx = np.outer(avg, avg)
            if self.use_bias:
                avg_mtx /= tlen * tlen
            else:
                avg_mtx /= tlen * (tlen - 1)
            # end if
            mtx -= avg_mtx
        # end if

        # Fix the average
        avg /= tlen

        return mtx, avg, tlen
    # end fix

# end SFACell
