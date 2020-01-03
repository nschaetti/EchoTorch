# -*- coding: utf-8 -*-
#
# File : echotorch/utils/conceptors/ConceptorSet.py
# Description : A class to store and manipulate a set of conceptors.
# Date : 6th of November, 2019
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

# Imports
import torch
from ..NeuralFilter import NeuralFilter
from .Conceptor import Conceptor
from echotorch.utils import quota, rank
from echotorch.utils.utility_functions import generalized_squared_cosine


# Set of conceptors : store and manipulate conceptors safely
class ConceptorSet(NeuralFilter):
    """
    Set of conceptors
    """
    # region BODY

    # Constructor
    def __init__(self, input_dim, *args, **kwargs):
        """
        Constructor
        :param args: Arguments
        :param kwargs: Positional arguments
        """
        # Super constructor
        super(ConceptorSet, self).__init__(
            input_dim=input_dim,
            output_dim=input_dim,
            *args,
            **kwargs
        )

        # Dimension
        self._conceptor_dim = input_dim
        self._current_conceptor_index = -1

        # We link conceptors to names
        self._conceptors = dict()
    # end __init__

    # region PROPERTIES

    # Access list of conceptors
    @property
    def conceptors(self):
        """
        Access list of conceptors
        :return: List of Conceptor object
        """
        return self._conceptors
    # end conceptors

    # Current selected conceptor
    @property
    def current_conceptor(self):
        """
        Current selected conceptor
        :return: Selected conceptor
        """
        return self.conceptors[self._current_conceptor_index]
    # end current_conceptor

    # Number of conceptors stored
    @property
    def count(self):
        """
        Number of conceptors stored
        :return: Number of conceptors stored (int)
        """
        return len(self._conceptors)
    # end count

    # endregion PROPERTIES

    # region PUBLIC

    # NOT A
    def N(self):
        """
        NOT A - Subspace not populated by conceptors
        :return: Matrix N (Conceptor)
        """
        return self.A().NOT()
    # end N

    # Conceptor matrix of NOT A
    def F(self):
        """
        NOT A - Conceptor matrix of subspace not populated by conceptors
        :return: Matrix N (Conceptor)
        """
        if self.is_null():
            return torch.eye(self.input_dim, dtype=self._dtype)
        else:
            return self.N().conceptor_matrix()
        # end if
    # end F

    # OR of all conceptors stored
    def A(self, tol=1e-14):
        """
        OR of all conceptors stored
        :return: OR (Conceptor) of all conceptors stored
        """
        # Start at 0
        A = Conceptor(input_dim=self._conceptor_dim, aperture=1, dtype=self._dtype)

        # For each conceptor
        for kc, C in self._conceptors.items():
            A.OR_(C, tol=tol)
        # end for

        return A
    # end A

    # Quota of the set of Conceptors
    def quota(self):
        """
        Quota of the set of Conceptors.
        The space taken by the Conceptors
        in the reservoir space.
        """
        if self.is_null():
            return 0.0
        else:
            return self.A().quota
        # end if
    # end quota

    # The set contains only zero null conceptors
    def is_null(self):
        """
        The set contains only zero null conceptors
        """
        for k, c in self.conceptors.items():
            if not c.is_null():
                return False
            # end if
        # end for
        return True
    # end is_null

    # Multiply aperture of each conceptor by a factor gamma
    def PHI(self, gamma):
        """
        Multiply aperture of each conceptor by a factor gamma
        :param gamma: Multiplicative factor
        """
        for conceptor_i in range(self.count):
            self.conceptors[conceptor_i].PHI(gamma)
        # end for
    # end PHI

    # Similarity between two conceptors
    def similarity(self, conceptor_i, conceptor_j, based_on='C', sim_func=generalized_squared_cosine):
        """
        Similarity between two conceptors
        :param conceptor_i: First conceptor index
        :param conceptor_j: Second coneptor index
        :param based_on: Similarity based on C ('C') or R ('R)
        :param sim_func: Simularity function (default: generalized squared cosine)
        :return: Similarity
        """
        return Conceptor.similarity(
            self.conceptors[conceptor_i],
            self.conceptors[conceptor_j],
            based_on=based_on,
            sim_func=sim_func
        )
    # end similarity

    # Compute similarity matrix between conceptors
    def similarity_matrix(self, based_on='C', sim_func=generalized_squared_cosine):
        """
        Compute similarity matrix between conceptors
        :param based_on: Similarity based on C ('C') or R ('R)
        :param sim_func: Similarity function (default: generalized squared cosine)
        :return: Similarity matrix as torch tensor
        """
        # Similarity matrix
        sim_matrix = torch.zeros(self.count, self.count)

        # For each pair of conceptor
        for i in range(self.count):
            for j in range(self.count):
                sim_matrix[i, j] = self.similarity(i, j, based_on, sim_func)
            # end for
        # end for
        return sim_matrix
    # end similarity_matrix

    # Intersection matrix between conceptors
    def intersection_matrix(self, return_rank=False, gamma=1):
        """
        Union matrix between conceptors
        :param return_rank: Rank of union
        """
        # Intersection matrix
        if return_rank:
            intersection_matrix = torch.zeros(self.count, self.count, dtype=torch.long)
        else:
            intersection_matrix = torch.zeros(self.count, self.count)
        # end if

        # For each pair
        for i, i_name in enumerate(self._conceptors.keys()):
            for j, j_name in enumerate(self._conceptors.keys()):
                # Compute
                E = Conceptor.operator_AND(self._conceptors[i_name], self._conceptors[j_name])

                # Change aperture
                if gamma != 1:
                    E.PHI(gamma)
                # end if

                # Compute rank or quota
                if return_rank:
                    intersection_matrix[i, j] = rank(E.C)
                else:
                    intersection_matrix[i, j] = quota(E.C)
                # end if
            # end for
        # end for

        return intersection_matrix
    # end intersection_matrix

    # Set conceptor index to use
    def set(self, k):
        """
        Set k index to use
        :param conceptor_i: Conceptor index to use
        """
        if k in self.conceptors.keys():
            self._current_conceptor_index = k
        else:
            raise Exception("Unknown conceptor {}".format(k))
        # end if
    # end set

    # Reset the set (empty list, reset A)
    def reset(self):
        """
        Reset the set
        """
        # Empty dict
        self._conceptors = dict()
    # end reset

    # Add a conceptor to the set
    def add(self, idx, c):
        """
        Add a conceptor to the set
        :param idx: Name associated with the conceptor
        :param c: Conceptor object
        """
        # Add
        self._conceptors[idx] = c
        self.add_trainable(c)
    # end add

    # Delete a conceptor from the set
    def delete(self, idx):
        """
        Delete a conceptor from the set
        :param idx: Index associated with the conceptor removed
        :return: New matrix A, the OR of all stored conceptors
        """
        # Remove
        self.remove_trainable(self._conceptors[idx])
        del self._conceptors[idx]
    # end delete

    # Morph conceptors in the set
    def morphing(self, morphing_vector):
        """
        Morph conceptors in the set
        """
        # Get morphed C
        Cm = self.morphed_C(morphing_vector)
        new_C = Conceptor(self._input_dim, aperture=1, dtype=self._dtype)
        new_C.set_C(Cm)
        return new_C
    # end morphing

    # Get morphed conceptor matrix
    def morphed_C(self, morphing_vector):
        """
        Get morphed conceptor matrix
        """
        # Start with zero
        Cm = torch.zeros(self.input_dim, self.input_dim, dtype=self._dtype)

        # For each conceptor
        for c_i, c_name in enumerate(self.conceptors.keys()):
            Cc = self.conceptors[c_name]
            Cm += morphing_vector[c_i] * Cc.C
        # end for

        return Cm
    # end morphed_C

    # Negative evidence for a Conceptor
    # TODO: Test
    def Eneg(self, conceptor_i, x, tol=1e-14):
        """
        Negative evidence
        :param conceptor_i: Index of the conceptor to compare to.
        :param x: Reservoir states
        :return: Evidence against
        """
        # Empty conceptor
        # Cothers = Conceptor.empty(self._conceptor_dim)

        # OR for all other Conceptor
        # for k, C in self.conceptors.items():
        #     if k != conceptor_i:
        #         Cothers = Conceptor.operator_OR(Cothers, C)
            # end if
        # end for

        # Start at 0
        others = Conceptor(input_dim=self._conceptor_dim, aperture=1, dtype=self._dtype)

        # For each conceptor
        for kc, C in self._conceptors.items():
            if kc != conceptor_i:
                others.OR_(C, tol=tol)
            # end if
        # end for

        # NOT others
        not_others = Conceptor.operator_NOT(others)

        # Tensor dim
        return Conceptor.evidence(not_others, x)
    # end Eneg

    # Positive evidence for a conceptor
    # TODO: Test
    def Eplus(self, conceptor_i, x):
        """
        Evidence for a Conceptor
        :param conceptor_i: Index of the conceptor to compare to.
        :param x: Reservoir states
        :return: Positive evidence
        """
        return self.conceptors[conceptor_i].E(x)
    # end Eplus

    # Total evidence for a Conceptor
    # TODO: Test
    def E(self, conceptor_i, x):
        """
        Total evidence for a Conceptor
        :param conceptor_i: Index of the conceptor to compare to.
        :param x: Reservoir states
        :return: Total evidence
        """
        # Total evidence
        return (self.Eplus(conceptor_i, x) + self.Eneg(conceptor_i, x)) / 2.0
    # end E

    # Get evidences for each conceptor
    # TODO: Test
    def evidences(self, x, based_on='both', average=False):
        """
        Get evidences for each conceptor
        :param x: Matrix of points in the reservoir space (T x Nx)
        :param based_on: both (positive evidence + negative evidence), positive, negative
        :return: Evidence matrix with evidence for each conceptor (T x conceptors)
        """
        # A single point in reservoir space
        if x.ndim == 2:
            # Time length
            time_length = x.size(0)

            # Evidence vector
            evidences_matrix = torch.zeros(time_length, self.count)

            # For each conceptor
            for c_i, k in enumerate(self.conceptors.keys()):
                # Evidences for this conceptor
                if based_on == 'both':
                    evidences_matrix[:, c_i] = self.E(k, x)
                elif based_on == 'positive':
                    evidences_matrix[:, c_i] = self.Eplus(k, x)
                elif based_on == 'negative':
                    evidences_matrix[:, c_i] = self.Eneg(k, x)
                else:
                    raise Exception("Waiting for both, positive or negative for based_on, got {}".format(based_on))
                # end if
            # end for

            # Average
            if average:
                return torch.mean(evidences_matrix, axis=0)
            else:
                return evidences_matrix
            # end if
        elif x.ndim == 3:
            raise Exception("Waiting for 2-dim tensor, got {} instead".format(x.ndim))
        #end if
    # end evidences

    # endregion PUBLIC

    # region PRIVATE

    # endregion PRIVATE

    # region OVERRIDE

    # Learn filter
    def filter_fit(self, X, *args, **kwargs):
        """
        Filter signal
        :param X: Signal to filter
        :return: Original signal
        """
        # Give vector to conceptor
        if self.count > 0:
            return self._conceptors[self._current_conceptor_index](X)
        else:
            raise Exception("Conceptor set is empty!")
        # end if

    # end filter_fit

    # Filter signal
    def filter_transform(self, X, *args, **kwargs):
        """
        Filter signal
        :param X: State to filter
        :param M: Morphing vector
        :return: Filtered signal
        """
        # Morphing vector present ?
        if "morphing_vector" in kwargs.keys():
            # Morphing vector
            morphing_vector = kwargs["morphing_vector"]

            # Morph conceptors
            Cm = self.morphed_C(morphing_vector)

            # Filter with morphed conceptor
            return Cm.mv(X)
        else:
            return self.current_conceptor(X)
        # end if
    # end filter_transform

    # Extra-information
    def extra_repr(self):
        """
        Extra-information
        :return: String
        """
        s = super(ConceptorSet, self).extra_repr()
        s += ', count=' + str(self.count) + ', current={_current_conceptor_index}, conceptors={_conceptors}'
        return s.format(**self.__dict__)
    # end extra_repr

    # Get item
    def __getitem__(self, item):
        """
        Get item
        """
        return self._conceptors[item]
    # end __getitem__

    # Set item
    def __setitem__(self, key, value):
        """
        Set item
        """
        self._conceptors[key] = value
    # end __setitem__

    # endregion OVERRIDE

# end ConceptorSet
