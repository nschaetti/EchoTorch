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
import torch
from typing import Union, List
import echotorch.nn.conceptors
from echotorch.nn.conceptors import Conceptor, ConceptorSet
from echotorch.utils.utility_functions import generalized_squared_cosine


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
def csim(
        c1: Union[Conceptor, List[Conceptor], ConceptorSet],
        c2: Union[Conceptor, List[Conceptor], ConceptorSet],
        based_on: str = 'C',
        sim_func: callable = generalized_squared_cosine
):
    """
    Conceptor similarity based on generalized square cosine
    """
    # Types
    c1_list = isinstance(c1, list)
    c2_list = isinstance(c2, list)

    if not c1_list and not c2_list:
        return c1.sim(c2, based_on, sim_func)
    elif not c1_list and c2_list:
        return c1.sim(c2, based_on, sim_func)
    elif c1_list and not c2_list:
        return c2.sim(c1, based_on, sim_func)
    else:
        sim_matrix = torch.zeros(len(c1), len(c2))
        for c1_i, c1_c in enumerate(c1):
            for c2_i, c2_c in enumerate(c2):
                sim_matrix[c1_i, c2_i] = echotorch.nn.Conceptor.similarity(
                    c1_c,
                    c2_c,
                    based_on,
                    sim_func
                )
            # end for
        # end for
        return sim_matrix
    # end if
# end csim


# conceptor similarity
def csimilarity(
        c1: Union[Conceptor, List[Conceptor], ConceptorSet],
        c2: Union[Conceptor, List[Conceptor], ConceptorSet],
        based_on: str = 'C',
        sim_func: callable = generalized_squared_cosine
):
    """
    Conceptor similarity
    """
    if isinstance(c1, Conceptor) and isinstance(c1, Conceptor):
        return echotorch.nn.Conceptor.similarity(c1, c2, based_on, sim_func)
    elif isinstance(c1, ConceptorSet) and isinstance(c2, Conceptor) or \
            isinstance(c1, Conceptor) and isinstance(c2, list):
        return c1.sim(c2, based_on, sim_func)
    elif isinstance(c1, Conceptor) and isinstance(c2, ConceptorSet) or \
            isinstance(c1, list) and isinstance(c2, Conceptor):
        return c2.sim(c1, based_on, sim_func)
    elif isinstance(c1, list) and isinstance(c2, list):
        sim_matrix = torch.zeros(len(c1), len(c2))
        for c1_i, c1_c in c1:
            for c2_i, c2_c in c2:
                sim_matrix[c1_i, c2_i] = Conceptor.similarity(
                    c1_c,
                    c2_c,
                    based_on,
                    sim_func
                )
            # end for
        # end for
        return sim_matrix
    # end if
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
