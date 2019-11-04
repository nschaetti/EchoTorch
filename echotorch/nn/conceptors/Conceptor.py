# -*- coding: utf-8 -*-
#
# File : echotorch/nn/conceptors/Conceptor.py
# Description : Base conceptor class
# Date : 4th of November, 2019
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
Created on 4 November 2019
@author: Nils Schaetti
"""

# Imports
import torch
from torch.autograd import Variable
import math

from ..Node import Node


# Conceptor base class
class Conceptor(Node):
    """
    Conceptor base class
    """

    # Constructor
    def __init__(self, input_dim, aperture, C=None, R=None, *args, **kwargs):
        """
        Constructor
        """
        # Superclass
        super(Conceptor, self).__init__(
            input_dim=input_dim,
            output_dim=input_dim,
            *args,
            **kwargs
        )

        # Parameters
        self._aperture = aperture
        self._n_samples = 0
        c_size = input_dim

        # Initialize correlation matrix
        if R is None:
            self.register_buffer('R', Variable(torch.zeros(c_size, c_size, dtype=self._dtype), requires_grad=False))
        else:
            self.register_buffer('R', R)
            self._update_conceptor_matrix()
        # end if

        # Initialize conceptor matrix
        if C is None:
            self.register_buffer('C', Variable(torch.zeros(c_size, c_size, dtype=self._dtype), requires_grad=False))
        else:
            self.register_buffer('C', C)
            self._update_correlation_matrix()
        # end if
    # end __init__

    ######################
    # PROPERTIES
    ######################

    # Get aperture
    @property
    def aperture(self):
        """
        Get aperture
        :return: Aperture
        """
        return self._aperture
    # end aperture

    # Change aperture
    @aperture.setter
    def aperture(self, ap):
        """
        Change aperture
        """
        self._aperture = ap
        self._update_conceptor_matrix()
    # end aperture

    # Dimension
    @property
    def dim(self):
        """
        Dimension
        :return: Conceptor dimension
        """
        return self.C.size(0)
    # end dim

    ######################
    # PUBLIC
    ######################

    # Forward
    def forward(self, X):
        """
        Forward
        :param X: Reservoir states
        """
        # Not training
        if self.training:
            # Increment correlation matrices
            self._increment_correlation_matrices(X)
        else:
            return torch.mm(X, self.C)
        # end if

        return X
    # end forward

    # Reset
    def reset(self):
        """
        Reset
        :return:
        """
        # No samples
        self._n_samples = 0
        self.R.fill_(0.0)
    # end reset

    # Set correlation matrix
    def set_R(self, R):
        """
        Set correlation matrix
        :param R: Correlation matrix
        """
        self.R = R
        self.input_dim = R.size(0)
        self.output_dim = R.size(0)
        self._update_conceptor_matrix()
    # end set_R

    # Set conceptor matrix
    def set_C(self, C, aperture):
        """
        Set conceptor matrix
        :param C: Conceptor matrix
        :param aperture: Conceptor's aperture
        """
        self.C = C
        self.input_dim = C.size(0)
        self.output_dim = C.size(0)
        self._aperture = aperture
        self._update_correlation_matrix()
    # end set_C

    # Multiply aperture by a factor
    def PHI(self, gamma):
        """
        Multiply aperture by a factor
        :param gamma: Multiply aperture by a factor.
        """
        # Dimension
        dim = self.dim

        # Multiply by 0
        if gamma == 0:
            (U, S, V) = torch.svd(self.C)
            Sdiag = S
            Sdiag[Sdiag < 1] = torch.zeros((sum(Sdiag < 1), 1))
            Cnew = torch.mm(U, torch.mm(torch.diag(Sdiag), U.t()))
        elif gamma == float("inf"):
            (U, S, V) = torch.svd(self.C)
            Sdiag = S
            Sdiag[Sdiag > 0] = torch.ones(sum(Sdiag > 0), 1)
            Cnew = torch.mm(U, torch.mm(torch.diag(Sdiag), U.t()))
        else:
            Cnew = torch.mm(self.C, torch.inverse(self.C + math.pow(gamma, -2) * (torch.eye(dim) - self.C)))
        # end

        # Set aperture and C
        self.C = Cnew
        self._aperture *= gamma
    # end PHI

    # AND in Conceptor Logic
    def AND(self, B):
        """
        AND in Conceptor Logic
        :param B: Second conceptor operand (reservoir size x reservoir size)
        :return: Self AND B
        """
        # Dimension
        dim = self.dim()
        tol = 1e-14

        # Conceptor matrices
        Cc = self.C
        Bc = B.C

        # Apertures
        C_aperture = self.aperture
        B_aperture = B.aperture

        # SV on both conceptor
        (UC, SC, UtC) = torch.svd(Cc)
        (UB, SB, UtB) = torch.svd(Bc)

        # Get singular values
        dSC = SC
        dSB = SB

        # How many non-zero singular values
        numRankC = int(torch.sum(1.0 * (dSC > tol)))
        numRankB = int(torch.sum(1.0 * (dSB > tol)))

        # Select zero singular vector
        UC0 = UC[:, numRankC:]
        UB0 = UB[:, numRankB:]

        # SVD on UC0 + UB0
        # (W, Sigma, Wt) = lin.svd(np.dot(UC0, UC0.T) + np.dot(UB0, UB0.T))
        (W, Sigma, Wt) = torch.svd(UC0 @ UC0.t() + UB0 @ UB0.t())

        # Number of non-zero SV
        numRankSigma = int(sum(1.0 * (Sigma > tol)))

        # Select zero singular vector
        Wgk = W[:, numRankSigma:]

        # C and B
        # Wgk * (Wgk^T * (C^-1 + B^-1 - I) * Wgk)^-1 * Wgk^T
        CandB = Wgk @ torch.inverse(Wgk.t() @ (torch.pinverse(Cc, tol) + torch.pinverse(Bc, tol) - torch.eye(dim)) @ Wgk) @ Wgk.t()

        # New conceptor
        new_conceptor = Conceptor(
            input_dim=dim,
            aperture=1.0 / math.sqrt(math.pow(C_aperture, -2) + math.pow(B_aperture, -2)),
            C=CandB
        )
        return new_conceptor
    # end AND

    # AND in Conceptor Logic
    def AND_(self, B):
        """
        AND in Conceptor Logic
        :param B: Second conceptor operand (reservoir size x reservoir size)
        """
        # C AND B
        CandB = self.AND(B)
        self.set_C(CandB.C, CandB.aperture)
    # end AND_

    # OR in Conceptor Logic
    def OR(self, Q):
        """
        OR in Conceptor Logic
        :param Q: Second conceptor operand (reservoir size x reservoir size)
        :return: Self OR Q
        """
        # R OR Q
        return (self.NOT().AND(Q.NOT())).NOT()
    # end OR

    # OR in Conceptor Logic (in-place)
    def OR_(self, Q):
        """
        OR in Conceptor Logic (in-place)
        :param Q: Second operand Conceptor
        """
        newC = self.OR(Q)
        self.R = newC.R
        self._aperture = newC.aperture
        self.C = newC.C
    # end OR_

    # NOT
    def NOT(self):
        """
        NOT
        :return: ~C
        """
        # NOT correlation matrix
        not_R = torch.eye(self.input_dim) - self.R

        # New conceptor
        new_conceptor = Conceptor(
            input_dim=self.input_dim,
            aperture=1.0 / self._aperture,
            R=not_R
        )
        return new_conceptor
    # end NOT

    # NOT (in-place)
    def NOT_(self):
        """
        NOT (in-place)
        """
        self._aperture = 1.0 / self._aperture
        self.set_R(torch.eye(self.input_dim) - self.R)
    # end NOT_

    ######################
    # PRIVATE
    ######################

    # Update conceptor matrix
    def _update_conceptor_matrix(self):
        """
        Update conceptor matrix
        """
        # Normalize
        self.R /= self._n_samples

        # Compute SVD on R
        U, S, V = torch.svd(self.R)

        # Compute new singular values in the unit circle
        Snew = torch.mm(S, torch.inverse(S + math.pow(self._aperture, -2) * torch.eye(self.input_dim)))

        # Compute conceptor matrix
        self.C.data = torch.mm(torch.mm(U, Snew), U.t()).data
    # end _update_conceptor_matrix

    # Update correlation matrix
    def _update_correlation_matrix(self):
        """
        Update correlation matrix
        """
        self.R = math.pow(self._aperture, -2) * torch.mm(self.C, torch.inverse(torch.eye(self.dim) - self.C))
    # end _update_correlation_matrix

    # Increment correlation matrices
    def _increment_correlation_matrices(self, X):
        """
        Increment correlation matrices
        :param X: Reservoir states
        """
        # Learn length
        learn_length = X.size(1)

        # CoRrelation matrix of reservoir states
        self.R += (X.t() @ X) / float(learn_length)

        # Inc. n samples
        self._n_samples += 1

        # Update conceptor matrix
        self._update_conceptor_matrix()
    # end _increment_correlation_matrices

    ######################
    # STATIC
    ######################

    # NOT operator
    @staticmethod
    def operator_NOT(C):
        """
        NOT operator
        :param C: Conceptor matrix
        :return: NOT version of R
        """
        return C.NOT()
    # end operator_NOT

    # OR in Conceptor Logic
    @staticmethod
    def operator_OR(C, B):
        """
        OR in Conceptor Logic
        :param C: First Conceptor operand (reservoir size x reservoir size)
        :param B: Second Conceptor operand (reservoir size x reservoir size)
        :return: C OR B
        """
        # C OR B
        return C.OR(B)
    # end operator_OR

    # AND in Conceptor Logic
    @staticmethod
    def operator_AND(C, B):
        """
        AND in Conceptor Logic
        :param C: First Conceptor operand
        :param B: Second Conceptor operand
        :return: C AND B
        """
        # C AND B
        return C.AND(B)
    # end operator_AND

# end Conceptor
