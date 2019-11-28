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


# Set of conceptors : store and manipulate conceptors safely
class ConceptorSet(NeuralFilter):
    """
    Set of conceptors
    """

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

    #################
    # PROPERTIES
    #################

    # OR of all conceptors stored
    @property
    def A(self):
        """
        OR of all conceptors stored
        :return: OR (Conceptor) of all conceptors stored
        """
        # Start at 0
        A = Conceptor(input_dim=self._conceptor_dim, aperture=1)

        # For each conceptor
        for C in self._conceptors:
            A.OR_(C)
        # end for

        return A
    # end A

    # NOT A
    @property
    def N(self):
        """
        NOT A - Subspace not populated by conceptors
        :return: Matrix N (Conceptor)
        """
        return self._A.NOT()
    # end N

    # Access list of conceptors
    @property
    def conceptors(self):
        """
        Access list of conceptors
        :return: List of Conceptor object
        """
        return self._conceptors
    # end conceptors

    # Number of conceptors stored
    @property
    def count(self):
        """
        Number of conceptors stored
        :return: Number of conceptors stored (int)
        """
        return len(self._conceptors)
    # end count

    #################
    # PUBLIC
    #################

    # Set conceptor index to use
    def set(self, conceptor_i):
        """
        Set conceptor index to use
        :param conceptor_i: Conceptor index to use
        """
        self._current_conceptor_index = conceptor_i
    # end set

    # Learn filter
    def filter_fit(self, X):
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
    def filter_transform(self, X, M):
        """
        Filter signal
        :param X: State to filter
        :param M: Morphing vector
        :return: Filtered signal
        """
        return X
    # end filter_transform

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
    # end add

    # Delete a conceptor from the set
    def delete(self, name):
        """
        Delete a conceptor from the set
        :param name: Name associated with the conceptor removed
        :return: New matrix A, the OR of all stored conceptors
        """
        # Remove
        del self._conceptors[name]
    # end delete

    # Negative evidence for a Conceptor
    # TODO: Test
    def Eneg(self, conceptor_i, x):
        """
        Negative evidence
        :param conceptor_i: Index of the conceptor to compare to.
        :param x: Reservoir states
        :return: Evidence against
        """
        # Empty conceptor
        Cothers = Conceptor.empty(self._conceptor_dim)

        # OR for all other Conceptor
        for i, C in enumerate(self.conceptors):
            Cothers = Conceptor.operator_OR(Cothers, C)
        # end for

        # NOT others
        Nothers = Conceptor.operator_NOT(Cothers)

        return Conceptor.evidence(Nothers, x)
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
        # Target conceptor
        target_conceptor = self.conceptors[conceptor_i]

        # Total evidence
        return self.Eplus(conceptor_i, x) + self.Eneg(conceptor_i, x)
    # end E

    ###################
    # PRIVATE
    ###################

    ###################
    # OVERRIDE
    ###################

# end ConceptorSet
