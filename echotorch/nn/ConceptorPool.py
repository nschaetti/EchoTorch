# -*- coding: utf-8 -*-
#
# File : echotorch/nn/ConceptorPool.py
# Description : A pool of conceptor.
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
import math
from echotorch.utils import generalized_squared_cosine
import math as m
from torch.autograd import Variable
from .Conceptor import Conceptor


# Conceptor
class ConceptorPool(object):
    """
    ConceptorPool
    """

    # Contructor
    def __init__(self, conceptor_dim, conceptors=list(), esn=None, dtype=torch.float32):
        """
        Constructor
        :param conceptors:
        """
        # Properties
        self.conceptor_dim = conceptor_dim
        self.conceptors = conceptors
        self.name_to_conceptor = {}
        self.esn = esn
        self.dtype = dtype
    # end __init__

    ###############################################
    # PROPERTIES
    ###############################################

    # Singular values of A
    @property
    def A_SV(self):
        """
        Singular values of A
        :return:
        """
        return ConceptorPool.compute_A_SV(self.conceptors)
    # end A_SV

    # A (OR of all conceptors)
    @property
    def A(self):
        return ConceptorPool.compute_A(self.conceptors)
    # end A

    # Quota
    @property
    def quota(self):
        """
        Quota
        :return:
        """
        return ConceptorPool.compute_quota(self.conceptors)
    # end quota

    ###############################################
    # PUBLIC
    ###############################################

    # Get similarity matrix
    def similarity_matrix(self):
        """
        Get similarity matrix
        :return:
        """
        return ConceptorPool.compute_similarity_matrix(self.conceptors)
    # end similarity_matrix

    # Finalize conceptor
    def finalize_conceptor(self, i):
        """
        Finalize conceptor
        :param i:
        :return:
        """
        # Finalize
        self.conceptors[i].finalize()
    # end finalize_conceptor

    # Finalize all conceptors
    def finalize(self):
        """
        Finalize all conceptors
        :return:
        """
        for c in self.conceptors:
            c.finalize()
        # end for
    # end finalize

    # Positive evidence
    def E_plus(self, p):
        """
        Positive evidence
        :param x: states (x)
        :return:
        """
        return ConceptorPool.compute_E_plus(self.conceptors, self.esn, p)
    # end E_plus

    # Evidence for other
    def E_other(self, p):
        """
        Evidence for other
        :param p:
        :return:
        """
        # Sizes
        batch_size = p.shape[0]
        time_length = p.shape[1]

        # List of evidences
        evidences = torch.zeros(batch_size, time_length)

        # Compute hidden states
        x = self.esn(u=p, return_states=True)

        # All conceptors
        A = self.A

        # Not A
        N = A.logical_not()

        # For each batch
        for b in range(batch_size):
            # Time
            for t in range(time_length):
                evidences[b, t] = torch.mm(x[b, t].view(1, -1), N.get_C()).mm(x[b, t].view(-1, 1))
            # end for
        # end for
        return torch.mean(evidences, dim=1)
    # end E_other

    # Negative evidence
    def E_neg(self, p):
        """
        Negative evidence
        :param p:
        :return:
        """
        # Sizes
        batch_size = p.shape[0]
        n_conceptors = len(self.conceptors)
        time_length = p.shape[1]

        # List of evidences
        evidences = torch.zeros(batch_size, time_length, n_conceptors)

        # Compute hidden states
        x = self.esn(u=p, return_states=True)

        # For each batch
        for b in range(batch_size):
            # Time
            for i, c in enumerate(self.conceptors):
                # List of all conceptor without c
                other_c = list(self.conceptors)
                other_c.remove(c)

                # Compute A
                A = ConceptorPool.compute_A(other_c)

                # Not A
                N = A.logical_not()

                # For each conceptors
                for t in range(time_length):
                    evidences[b, t, i] = torch.mm(x[b, t].view(1, -1), N.get_C()).mm(
                        x[b, t].view(-1, 1))
                # end for
            # end for
        # end for
        return torch.mean(evidences, dim=1)
    # end E_neg

    # Evidence for each conceptor
    def E(self, p):
        """
        Evidence for each conceptor
        :return:
        """
        return (self.E_plus(p) + self.E_neg(p)) / 2.0
    # end E

    # New conceptor
    def add(self, aperture, name):
        """
        New conceptor
        :param aperture: Aperture
        :param name: Conceptor's name
        :return: New conceptor
        """
        new_conceptor = Conceptor(self.conceptor_dim, aperture=aperture, name=name, dtype=self.dtype)
        self.conceptors.append(new_conceptor)
        self.name_to_conceptor[name] = new_conceptor
        return new_conceptor
    # end add

    # Add an OR between conceptors
    def add_or(self, i, j):
        """
        Add an OR between conceptors
        :param i:
        :param j:
        :return:
        """
        self.append(self.conceptors[i].logical_or(self.conceptors[j]))
    # end for

    # Add an AND between conceptors
    def add_and(self, i, j):
        """
        Add an AND between conceptors
        :param i:
        :param j:
        :return:
        """
        self.append(self.conceptors[i].logical_and(self.conceptors[j]))
    # end for

    # Add a NOT of a conceptor
    def add_not(self, i):
        """
        Add an OR between conceptors
        :param i:
        :return:
        """
        self.append(self.conceptors[i].logical_not())
    # end for

    # Add (C1 OR ... OR CN)
    def add_A(self):
        """
        Add an OR between conceptors
        :param i:
        :return:
        """
        # Compute A
        A = ConceptorPool.compute_A(self.conceptors)

        # Append
        self.append(A)
    # end for

    # Add NOT (C1 OR ... OR CN)
    def add_Not_A(self):
        """
        Add an OR between conceptors
        :param i:
        :return:
        """
        # Compute A
        A = ConceptorPool.compute_A(self.conceptors)

        # Not A
        N = A.logical_not()

        # Append
        self.append(N)
    # end for

    # Append a conceptor
    def append(self, c):
        """
        Append a conceptor
        :param c:
        :return:
        """
        self.conceptors.append(c)
        self.name_to_conceptor[c.name] = c
    # end append

    # Morphing patterns
    def morphing(self, mu):
        """
        Morphing pattern
        :param conceptor_list:
        :return:
        """
        # For each conceptors
        for i, c in enumerate(self.conceptors):
            if i == 0:
                M = c.mul(mu[i])
            else:
                M += c.mul(mu[i])
                # end if
        # end for
        return M
    # end for

    ###############################################
    # PRIVATE
    ###############################################

    # Get item
    def __getitem__(self, item):
        """
        Get item
        :param item:
        :return:
        """
        if type(item) is int:
            return self.conceptors[item]
        elif type(item) is str:
            return self.name_to_conceptor[item]
        # end if
    # end __getitem__

    # Set item
    def __setitem__(self, key, value):
        """
        Set item
        :param key:
        :param value:
        :return:
        """
        self.conceptors[key] = value
    # end __setitem__

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return len(self.conceptors)
    # end __len__

    ###############################################
    # STATIC
    ###############################################

    # Get similarity matrix
    @staticmethod
    def compute_similarity_matrix(conceptors):
        """
        Get similarity matrix
        :return:
        """
        # Similarity matrix
        sim_matrix = torch.zeros(len(conceptors), len(conceptors))
        for i, ca in enumerate(conceptors):
            for j, cb in enumerate(conceptors):
                sim_matrix[i, j] = ca.sim(cb)
            # end for
        # end for
        return sim_matrix
    # end similarity_matrix

    # Positive evidence
    @staticmethod
    def compute_E_plus(conceptors, esn, p):
        """
        Positive evidence
        :param x: states (x)
        :return:
        """
        # Sizes
        batch_size = p.shape[0]
        n_conceptors = len(conceptors)
        time_length = p.shape[1]

        # List of evidences
        evidences = torch.zeros(batch_size, time_length, n_conceptors)

        # Compute hidden states
        x = esn(u=p, return_states=True)

        # For each batch
        for b in range(batch_size):
            # Time
            for t in range(time_length):
                # For each conceptors
                for i, c in enumerate(conceptors):
                    evidences[b, t, i] = torch.mm(x[b, t].view(1, -1), c.get_C()).mm(x[b, t].view(-1, 1))
                # end for
            # end for
        # end for
        return torch.mean(evidences, dim=1)
    # end E_plus

    # Get singular values of A
    @staticmethod
    def compute_A_SV(conceptors):
        """
        Get singular values of A
        :param conceptors:
        :return:
        """
        # A (OR of all conceptors)
        A = ConceptorPool.compute_A(conceptors)

        # Compute SVD
        _, S, _ = torch.svd(A.get_C())

        return S
    # end compute_A_SV

    # Compute A (OR of all conceptors
    @staticmethod
    def compute_A(conceptors):
        """
        Compute A (OR of all conceptors)
        :param conceptors:
        :return:
        """
        # OR for all conceptor
        for i, c in enumerate(conceptors):
            if i == 0:
                A = c
            else:
                A = c.logical_or(A)
            # end if
        # end for
        A.name = "A"

        return A
    # end compute_A

    # Compute quota
    @staticmethod
    def compute_quota(conceptors):
        """
        Compute quota
        :param conceptors:
        :return:
        """
        # Singular values of A
        S = ConceptorPool.compute_A_SV(conceptors)

        # Quota is the sum of SV
        return float(
            torch.mean(S)
        )
    # end compute_quota

# end __ConceptorPool__
