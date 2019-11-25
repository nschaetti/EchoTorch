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


# Set of conceptors : store and manipulate conceptors safely
class ConceptorSet(NeuralFilter):
    """
    Set of conceptors
    """

    # Constructor
    def __init__(self, *args, **kwargs):
        """
        Constructor
        :param args: Arguments
        :param kwargs: Positional arguments
        """
        # Super constructor
        super(ConceptorSet, self).__init__(
            args,
            kwargs
        )

        # Dimension
        self._conceptor_dim = self._input_dim

        # We link conceptors to names
        self.conceptors = dict()

        # OR of all conceptors stored
        self._reset_A()
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
        return self._A
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

    # Learn filter
    def filter_fit(self, X, Cn):
        """
        Filter signal
        :param X: Signal to filter
        :param Cn: Conceptor index
        :return: Original signal
        """
        #

        # Give vector to conceptor
        return self.conceptors[Cn](X)
    # end filter_fit

    # Filter signal
    def filter_transform(self, X, M):
        """
        Filter signal
        :param X: Signal to filter
        :param M: Morphing signal
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
        self.conceptors = dict()

        # Reset A
        self._reset_A()
    # end reset

    # Add a conceptor to the set
    def add(self, name, c):
        """
        Add a conceptor to the set
        :param name: Name associated with the conceptor
        :param c: Conceptor object
        :return: New matrix A, the OR of all stored conceptors
        """
        # Add
        self.conceptors[name] = c

        # Init A if needed
        if self.count == 1:
            self._A = c
        else:
            self._A.OR_(c)
        # end if

        return self.A
    # end add

    # Delete a conceptor from the set
    def delete(self, name):
        """
        Delete a conceptor from the set
        :param name: Name associated with the conceptor removed
        :return: New matrix A, the OR of all stored conceptors
        """
        # Conceptor matrix
        c = self.conceptors[name]

        # Remove
        del self.conceptors[name]

        # Recompute A
        if self.count == 0:
            self._reset_A()
        else:
            self._A.AND_(c.NOT())
        # end if

        return self.A
    # end delete

    ###################
    # PRIVATE
    ###################

    # Reset matrix A
    def _reset_A(self):
        """
        Reset matrix A
        """
        self._A = torch.zeros(self._conceptor_dim, self._conceptor_dim)
    # end _reset_A

    ###################
    # OVERRIDE
    ###################

# end ConceptorSet
