# -*- coding: utf-8 -*-
#
# File : echotorch/utils/conceptors/ConceptorBuilder.py
# Description : Utility class to create and build Conceptor object.
# Date : 28th of November, 2019
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
from echotorch.nn.conceptors import Conceptor


# Utility class to create and build Conceptor objects
class ConceptorBuilder:
    """
    Utility class to create and build Conceptor objects
    """

    # Constructor
    def __init__(self, conceptor_dim):
        """
        Constructor
        :param conceptor_dim: Conceptor dimension
        """
        self._conceptor_dim = conceptor_dim
    # end __init__

    ###############
    # PUBLIC
    ###############

    # Create empty conceptor
    def create_empty(self, incremental=False):
        """
        Create empty conceptor
        :param incremental: Create an incremental conceptor ?
        :return: Empty conceptor (zero)
        """
        return Conceptor.empty(self._conceptor_dim)
    # end create_empty_conceptor

    # Create conceptor from C matrix
    def create_from_C(self, C, aperture, incremental=False):
        """
        Create conceptor from C matrix
        :param C: C matrix
        :param aperture: Aperture associated with C
        :param incremental:
        :return: Conceptor object
        """
        new_conceptor = Conceptor.empty(self._conceptor_dim)
        new_conceptor.set_C(C, aperture)
        return new_conceptor
    # end create_from_C

    # Create conceptor from R correlation matrix
    def create_from_R(self, R):
        """
        Create conceptor from R correlation matrix
        :param R: Correlation matrix
        :return: Conceptor object
        """
        new_conceptor = Conceptor.empty(self._conceptor_dim)
        new_conceptor.set_R(R)
        return new_conceptor
    # Create from_from_R

# end ConceptorBuilder
