# -*- coding: utf-8 -*-
#
# File : papers/schaetti2016/transforms/Concat.py
# Description : Transform images to a concatenation of multiple transformations.
# Date : 11th of November, 2019
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
from PIL import Image
from ..Transformer import Transformer


# Return the same image
class Identity(Transformer):
    """
    Return the same image
    """

    # Constructor
    def __init__(self):
        """
        Constructor
        """
        # Super constructor
        super(Identity, self).__init__(
            input_dim=0,
            output_dim=0
        )
    # end __init__

    # Called to transform images
    def __call__(self, img):
        """
        Called to transform images
        :param img: Image to transform
        :return: Transformed images
        """
        return img
    # end __call__

    # Representation
    def __repr__(self):
        """
        Representation
        :return:
        """
        format_string = self.__class__.__name__
        return format_string
    # end __repr__

# end Identity
