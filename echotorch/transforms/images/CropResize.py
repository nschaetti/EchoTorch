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
import numpy as np
import torchvision.transforms.functional as F
from ..Transformer import Transformer


# Image crop and resize
class CropResize(Transformer):
    """
    Crop and resize images
    """

    # Constructor
    def __init__(self, size):
        """
        Constructor
        :param size: New image size
        """
        # Super constructor
        super(CropResize, self).__init__(
            input_dim=0,
            output_dim=0
        )

        # Properties
        self._size = size
    # end __init__

    ##################
    # PRIVATE
    ##################

    # Compute image border
    def _image_borders(self, img):
        """
        Compute image border
        :param img: Image
        :return: Position of borders
        """
        # Image size
        width, height = img.size

        # Image array
        img_array = np.array(img)

        # Positions
        height_start = 0
        height_end = 0
        width_start = 0
        width_end = 0

        # From the top
        for y in range(height):
            if np.sum(img_array[y, :]) > 0:
                height_end = y
            # end if
        # end while

        # From the bottom
        for y in range(height - 1, 0, -1):
            if np.sum(img_array[y, :]) > 0:
                height_start = y
            # end if
        # end for

        # From the left
        for x in range(width):
            if np.sum(img_array[:, x]) > 0:
                width_end = x
            # end if
        # end for

        # From the right
        for x in range(width - 1, 0, -1):
            if np.sum(img_array[:, x]) > 0:
                width_start = x
            # end if
        # end for

        return height_start, height_end, width_start, width_end
    # end _image_borders

    ################
    # OVERRIDE
    ################

    # Called to transform images
    def __call__(self, img):
        """
        Called to transform images
        :param img: Image to resize
        :return: Image to resize
        """
        # Image size
        width, height = img.size

        # Borders
        hs, he, ws, we = self._image_borders(img)

        # Padding
        hs -= 1
        he += 2
        ws -= 1
        we += 2

        # Limit
        if hs < 0:
            hs = 0
        # end if

        # Upper limit
        if he >= height:
            he = height - 1
        # end if

        # Limit
        if ws < 0:
            ws = 0
        # end if

        # Upper limit
        if we >= width:
            we = width - 1
        # end if

        # Crop image
        cropped_image = img.crop((ws, hs, we, he))

        # Resize
        return F.resize(cropped_image, (self._size, self._size))
    # end __call__

# end CropResize
