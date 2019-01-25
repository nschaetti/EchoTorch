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
    def __init__(self, conceptor_dim, aperture=0.0, with_bias=True, learning_algo='inv', name=""):
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
    # end __init__

    ###############################################
    # PROPERTIES
    ###############################################

    ###############################################
    # PUBLIC
    ###############################################

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
        x = x.unsqueeze(0)
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

# end RRCell
