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
import echotorch.nn as etnn
from torch.autograd import Variable


# Compute the approximation of time derivative
def time_derivative(x, time_step):
    """
    Compute the approximation of time derivative
    :param x: Input multi-dimensional signal
    :return: Input derivative
    """
    return (x[1:, :] - x[:-1, :]) / time_step
# end time_derivative


# Slow feature analysis
class SFACell(etnn.Node):
    """
    Extract the slowly varying components from input data.
    """

    # Constructor
    def __init__(self, input_dim, output_dim, whitening=True, time_step=1.0, dtype=torch.float64):
        """
        Constructor
        :param input_dim: Input dimension
        :param output_dim: Number of slow feature to extract
        """
        # Check that output is not bigger
        # than the input dimension
        if output_dim > input_dim:
            raise Exception("Output dimension cannot be bigger than the input dimension : {} vs {}".format(
                input_dim,
                output_dim
            ))
        # end if

        # Call upper  class
        super(SFACell, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            dtype=dtype
        )

        # Compute the number of samples
        self._n_samples = 0.0
        self._whitening = whitening
        self._time_step = time_step

        # Create buffer for the covariance matrix
        self.register_buffer(
            'xTx',
            Variable(torch.zeros(self._input_dim, self._input_dim, dtype=dtype), requires_grad=False)
        )

        # Create buffer for the covariance matrix of the derivatives
        self.register_buffer(
            'dxTdx',
            Variable(torch.zeros(self._input_dim, self._input_dim, dtype=dtype), requires_grad=False)
        )

        # Create buffer for the trained project matrix
        self.register_buffer(
            'W',
            Variable(torch.zeros(output_dim, self._input_dim, dtype=dtype), requires_grad=False)
        )

        # Create buffer for the trained bias
        self.register_buffer(
            'b',
            Variable(torch.zeros(self._input_dim, dtype=dtype), requires_grad=False)
        )
    # end __init__

    # region PROPERTIES

    # endregion PROPERTIES

    # region PUBLIC

    # Forward
    def forward(self, x):
        """
        Forward
        :param x: Input signal.
        :return: Output or hidden states
        """
        # Dims
        batch_size = x.size(0)
        time_length = x.size(1)

        # If in training mode
        if self.training:
            # Compute bias
            self.b += -torch.mean(torch.mean(x, dim=1), dim=0)

            # Center x
            x += self.b

            # For each batch
            for b in range(batch_size):
                # Use whitening ?
                if self._whitening:
                    # Compute x covariance matrix
                    self.xTx += torch.mm(x[b].t(), x[b]) / time_length

                    # Eigen-decomposition of the covariance matrix
                    D, U = torch.eig(self.xTx, eigenvectors=True)

                    # Check eigenvalues
                    self._check_eigenvalues(D)

                    # Remove imaginary parts and compute the diagonal matrix
                    D = torch.diag(D[:, 0])

                    # Whitening matrix S
                    S = torch.mm(torch.sqrt(torch.inverse(D)), U.t())

                    # Project the input signal x
                    # Into the space with unite-covariance matrix
                    xs = torch.mm(S, x[b].t()).t()
                else:
                    xs = x[b]
                # end if

                # Compute time derivatives
                dxs = time_derivative(xs, self._time_step)

                # Increment the time derivatives covariance matrix
                self.dxTdx += torch.mm(dxs.t(), dxs) / time_length

                # Samples computed
                self._n_samples += 1.0
            # end for

            # Compute eigen decomposition
            L, V = torch.eig(self.dxTdx, eigenvectors=True)

            # Check eigenvalues
            self._check_eigenvalues(L)

            # Keep only the slowest
            Vs = self._slowest_features(L, V)

            # Update W
            self.W = torch.mm(
                Vs.t(),
                torch.mm(torch.sqrt(torch.inverse(D)), U.t())
            )
        # end if

        # Empty tensor for output
        outputs = torch.zeros(batch_size, time_length, self._output_dim, dtype=self._dtype)

        # For each batch
        for b in range(batch_size):
            # Compute outputs
            outputs[b] = torch.mm(self.W, (x[b] + self.b).t()).t()
        # end for

        return outputs
    # end forward

    # Finish training
    def finalize(self):
        """
        Finalize training with LU factorization or Pseudo-inverse
        """
        # Trained
        self.train(False)
    # end finalize

    # endregion PUBLIC

    # region PRIVATE

    # Keep only the slowest features
    def _slowest_features(self, L, V):
        """
        Keep only  the slowest features
        :param L: Eigenvalues
        :param V: Eigenvectors
        :return: The eigenvector corresponding to the smallest eigenvalues
        """
        return torch.index_select(V, 1, torch.argsort(L[:, 0])[:self._output_dim])
    # end _slowest_features

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
            raise Exception("Got negative eigenvalues: {}".format(wB))
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

    # Check eigenvalues
    def _check_eigenvalues(self, L):
        """
        Check eigenvalues
        :param L: Matrix (n x 2) with real (1) and imaginary (2) parts.
        """
        if not (torch.all(L[:, 1] == 0) and torch.all(L[:, 0] >= 0.0)):
            raise Exception("Error, got imaginary or negative eigenvalues for whitening : {}".format(D))
        # end if
    # end _check_eigenvalues

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
        :param mtx: The cumulated covariance matrix
        :param avg: The cumulated average
        :param tlen: The total length of samples
        :param center: True if average must be removed of the covariance matrix
        :return: Fixed covariance matrix, average and total length
        """
        # Divide the covariance matrix
        # by the total length of training
        # samples.
        if self.use_bias:
            mtx /= tlen
        else:
            mtx /= tlen - 1
        # end if

        # Remove the mean to the
        # covariance matrix
        if center:
            avg_mtx = torch.ger(avg, avg)
            if self.use_bias:
                avg_mtx /= tlen * tlen
            else:
                avg_mtx /= tlen * (tlen - 1)
            # end if
            mtx -= avg_mtx
        # end if

        # Compute the average activation
        # for each units.
        avg /= tlen

        return mtx, avg, tlen
    # end fix

    # endregion PRIVATE

# end SFACell
