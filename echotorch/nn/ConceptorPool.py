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
    def __init__(self, conceptors=list(), dtype=torch.float32):
        """
        Constructor
        :param conceptors:
        """
        # Properties
        self.conceptors = conceptors
        self.name_to_conceptor = {}
        self.dtype = dtype
    # end __init__

    ###############################################
    # PUBLIC
    ###############################################

    # Evidence for each conceptor
    def E(self):
        """
        Evidence for each conceptor
        :return:
        """
        pass
    # end E

    # New conceptor
    def add(self, conceptor_dim, aperture, name):
        """
        New conceptor
        :param conceptor_dim: Conceptor's dim
        :param aperture: Aperture
        :param name: Conceptor's name
        :return: New conceptor
        """
        new_conceptor = Conceptor(conceptor_dim, aperture=aperture, name=name, dtype=self.dtype)
        self.conceptors.append(new_conceptor)
        self.name_to_conceptor[name] = new_conceptor
        return new_conceptor
    # end add

    # Append a conceptor
    def append(self, c):
        """
        Append a conceptor
        :param c:
        :return:
        """
        self.conceptors.append(c)
        self.conceptors[c.name] = c
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

    # Get attribute
    def __getattr__(self, item):
        """
        Get attribute
        :param item:
        :return:
        """
        return self.name_to_conceptor[item]
    # end __getattr__

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return len(self.conceptors)
    # end __len__

# end __ConceptorPool__
