# -*- coding: utf-8 -*-
#
# File : echotorch/conceptors.py
# Description : EchoTorch conceptors creation and management utility functions.
# Date : 9th of April, 2021
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
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>,
# University of Geneva <nils.schaetti@unige.ch>


# Imports
import echotorch.nn.conceptors


# conceptor zero
def czero(input_dim: int) -> echotorch.nn.Conceptor:
    """
    Create an empty Conceptor (zero) of specific size
    """
    return echotorch.nn.conceptors.Conceptor.zero(input_dim)
# end czero


# conceptor one
def cone(input_dim: int) -> echotorch.nn.Conceptor:
    """
    Create an unit Conceptor of specific size
    """
    return echotorch.nn.conceptors.Conceptor.identity(input_dim)
# end cone


# conceptor one
def cidentity(input_dim: int) -> echotorch.nn.Conceptor:
    """
    Create an unit Conceptor of specific size
    """
    return cone(input_dim)
# end cidentity


# conceptor similarity
def csim(c1: echotorch.nn.Conceptor, c2: echotorch.nn.Conceptor, based_on='C'):
    """
    Conceptor similarity based on generalized square cosine
    """
    return echotorch.nn.Conceptor.sim(c1, c2, based_on)
# end csim


# conceptor similarity
def csimilarity(c1: echotorch.nn.Conceptor, c2: echotorch.nn.Conceptor, based_on='C'):
    """
    Conceptor similarity based on generalized square cosine
    """
    return echotorch.nn.Conceptor.sim(c1, c2, based_on)
# end csimilarity


# OR operator
def OR(c1: echotorch.nn.Conceptor, c2: echotorch.nn.Conceptor):
    """
    OR operator
    """
    return echotorch.nn.Conceptor.operator_OR(c1, c2)
# end OR


# AND operator
def AND(c1: echotorch.nn.Conceptor, c2: echotorch.nn.Conceptor):
    """
    AND operator
    """
    return echotorch.nn.Conceptor.operator_AND(c1, c2)
# end AND


# NOT operator
def NOT(c1: echotorch.nn.Conceptor):
    """
    NOT operator
    """
    return echotorch.nn.Conceptor.operator_NOT(c1)
# end NOT


# PHI operator
def PHI(c, gamma):
    """
    PHI operator
    :param c:
    :param gamma:
    :return:
    """
    return echotorch.nn.Conceptor.operator_PHI(c, gamma)
# end PHI


# Conceptor constructor
def conceptor(input_dim, aperture, *args, **kwargs):
    """
    Conceptor constructor
    """
    return echotorch.nn.Conceptor(
        input_dim,
        aperture,
        *args,
        **kwargs
    )
# end conceptor


# Conceptor set
def conceptor_set(input_dim, *args, **kwargs):
    """
    Conceptor set
    :param input_dim:
    :param args:
    :param kwargs:
    :return:
    """
    return echotorch.nn.conceptors.ConceptorSet(
        input_dim,
        *args,
        **kwargs
    )
# end conctor_set
