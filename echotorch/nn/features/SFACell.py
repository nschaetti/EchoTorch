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
from ..Node import Node
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
class SFACell(Node):
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

    # endregion PRIVATE

# end SFACell
