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
from .RRCell import RRCell
import math


# Conceptor
class Conceptor(RRCell):
    """
    Conceptor
    """

    # Constructor
    def __init__(self, conceptor_dim, aperture=0.0, with_bias=False, learning_algo='inv', name="", conceptor_matrix=None):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param output_dim: Reservoir size
        """
        super(Conceptor, self).__init__(conceptor_dim, conceptor_dim, ridge_param=aperture, feedbacks=False, with_bias=with_bias, learning_algo=learning_algo)

        # Properties
        self.conceptor_dim = conceptor_dim
        self.aperture = aperture
        self.name = name
        if conceptor_matrix is not None:
            self.w_out = conceptor_matrix
            self.train(False)
        # end if
    # end __init__

    ###############################################
    # PROPERTIES
    ###############################################

    # Compute quota
    @property
    def quota(self):
        """
        Compute quota
        :return:
        """
        # Conceptor matrix
        conceptor_matrix = self.get_C()

        # Compute sum of singular values devided by number of neurons
        return float(torch.sum(conceptor_matrix.mm(torch.eye(self.conceptor_dim))) / self.conceptor_dim)
    # end quota

    ###############################################
    # PUBLIC
    ###############################################

    # Change aperture
    def set_aperture(self, new_a):
        """
        Change aperture
        :param new_a:
        :return:
        """
        # Conceptor matrix
        c = self.w_out.clone()

        # New tensor
        self.w_out = c.mm(torch.inverse(c + torch.pow(new_a / self.aperture, -2) * (torch.eye(self.conceptor_dim) - c)))
    # end set_aperture

    # Output matrix
    def get_C(self):
        """
        Output matrix
        :return:
        """
        return self.w_out
    # end get_w_out

    # Finish training
    def finalize(self):
        """
        Finalize training with LU factorization or Pseudo-inverse
        """
        if self.learning_algo == 'inv':
            ridge_xTx = self.xTx + math.pow(self.ridge_param, -2) * torch.eye(self.input_dim + self.with_bias)
            inv_xTx = ridge_xTx.inverse()
            self.w_out.data = torch.mm(inv_xTx, self.xTy).data
        else:
            self.w_out.data = torch.gesv(self.xTy, self.xTx + torch.eye(self.esn_cell.output_dim).mul(self.ridge_param)).data
        # end if

        # Not in training mode anymore
        self.train(False)
    # end finalize

    # Set conceptor
    def set_conceptor(self, c):
        """
        Set conceptor
        :param c:
        :return:
        """
        # Set matrix
        self.w_out.data = c
    # end set_conceptor

    # Singular values
    def singular_values(self):
        """
        Singular values
        :return:
        """
        return self.w_out.diag()
    # end singular_values

    # Some of singular values
    def get_quota(self):
        """
        Sum of singular values
        :return:
        """
        return float(torch.sum(self.singular_values()))
    # end get_quota

    ###############################################
    # OPERATORS
    ###############################################

    # Positive evidence
    def E_plus(self, x):
        """
        Positive evidence
        :param x: states (x)
        :return:
        """
        return x.mm(self.w_out).mm(x.t())
    # end E_plus

    # Evidence against
    def E_neg(self, x, conceptor_list):
        """
        Evidence against
        :param x:
        :param conceptor_list:
        :return:
        """
        # For each conceptor in the list
        for i, c in enumerate(conceptor_list):
            if i == 0:
                new_c = c
            else:
                new_c = new_c.logical_or(c)
            # end if
        # end for

        # Take the not
        N = new_c.logical_not()

        return x.t().mm(N.w_out).mm(x)
    # end E_neg

    # Evidence
    def E(self, x, conceptor_list):
        """
        Evidence
        :param x:
        :param conceptor_list:
        :return:
        """
        return self.E_plus(x) + self.E_neg(x, conceptor_list)
    # end E

    # OR
    def logical_or(self, c):
        """
        Logical OR
        :param c:
        :return:
        """
        # New conceptor
        new_c = Conceptor(self.conceptor_dim)

        # Matrices
        C = self.w_out
        B = c.get_w_out()
        I = torch.eye(self.conceptor_dim)

        # Compute C1 \/ C2
        conceptor_matrix = torch.inverse(I + torch.inverse(C.mm(torch.inverse(I - C)) + B.mm(torch.inverse(I - B))))

        # Set conceptor
        new_c.set_conceptor(conceptor_matrix)

        return new_c
    # end logical_or

    # OR
    def __or__(self, other):
        """
        OR
        :param other:
        :return:
        """
        return self.logical_or(other)
    # end __or__

    # NOT
    def logical_not(self):
        """
        Logical NOT
        :param c:
        :return:
        """
        # New conceptor
        new_c = Conceptor(self.conceptor_dim)

        # Matrices
        C = self.w_out

        # Compute not C
        conceptor_matrix = torch.eye(self.conceptor_dim) - C

        # Set conceptor
        new_c.set_conceptor(conceptor_matrix)

        return new_c
    # end logical_not

    # Not
    def __invert__(self):
        """
        NOT
        :return:
        """
        return self.logical_not()
    # end __invert__

    # AND
    def logical_and(self, c):
        """
        Logical AND
        :param c:
        :return:
        """
        # New conceptor
        new_c = Conceptor(self.conceptor_dim)

        # Matrices
        C = self.w_out
        B = c.get_w_out()

        # Compute C1 /\ C2
        conceptor_matrix = torch.inverse(torch.inverse(C) + torch.inverse(B) + torch.eye(self.conceptor_dim))

        # Set conceptor
        new_c.set_conceptor(conceptor_matrix)

        return new_c
    # end logical_and

    # AND
    def __and__(self, other):
        """
        AND
        :param other:
        :return:
        """
        return self.logical_and(other)
    # end __and__

    # Greater or equal
    def __ge__(self, other):
        """
        Greater or equal
        :param other:
        :return:
        """
        # Compute eigenvalues of a - b
        eig_v = torch.eig(other.get_C() - self.w_out, eigenvectors=False)
        return float(torch.max(eig_v)) >= 0.0
    # end __ge__

    # Greater
    def __gt__(self, other):
        """
        Greater
        :param other:
        :return:
        """
        # Compute eigenvalues of a - b
        eig_v = torch.eig(other.get_C() - self.w_out, eigenvectors=False)
        return float(torch.max(eig_v)) > 0.0
    # end __gt__

    # Less
    def __lt__(self, other):
        """
        Less than
        :param other:
        :return:
        """
        return not self >= other
    # end __lt__

    # Less or equal
    def __le__(self, other):
        """
        Less or equal
        :param other:
        :return:
        """
        return not self > other
    # end __le__

# end RRCell
