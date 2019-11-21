# -*- coding: utf-8 -*-
#
# File : echotorch/transforms/ToOneHot.py
# Description : Transform integer targets to one-hot vectors.
# Date : 21th of November, 2019
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
from ..Transformer import Transformer


# Transform index to one-hot vector
class ToOneHot(Transformer):
    """
    Transform class index to one-hot vector
    """

    # Constructor
    def __init__(self, class_size):
        """
        Constructor
        :param class_size: Number of classes
        """
        # Super constructor
        super(ToOneHot, self).__init__(
            input_dim=0,
            output_dim=class_size
        )

        # Properties
        self.class_size = class_size

        # One-hot vectors
        self._one_hot_vectors = torch.eye(self.output_dim, dtype=self._dtype)
    # end __init__

    ######################
    # Properties
    ######################

    ######################
    # Override
    ######################

    ######################
    # Private
    ######################

    # Transform
    def _transform(self, idx):
        """
        Transform input
        :param idx: Index
        :return: Transformed to one-hot
        """
        return self._one_hot_vectors[idx]
    # end _transform

    ##############################################
    # Static
    ##############################################

# end ToOneHot
