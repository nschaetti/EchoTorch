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


# Concat several transformations together
class Concat(Transformer):
    """
    Concat several transformation together.
    """

    # Constructor
    def __init__(self, transforms, sequential=False):
        """
        Constructor
        :param transforms: Transforms to concat
        :param sequential: If True, each transform takes its inputs from previous transformer
        """
        # Super constructor
        super(Concat, self).__init__(
            input_dim=0,
            output_dim=0
        )

        # Properties
        self.transforms = transforms
        self.sequential = sequential
    # end __init__

    # Called to transform images
    def __call__(self, img):
        """
        Called to transform images
        :param img: Image to transform
        :return: Transformed images
        """
        # New image
        new_img = None
        past_img = None

        # For each transformations
        for t_i, t in enumerate(self.transforms):
            # Transform image
            if self.sequential and t_i > 0:
                trans_img = t(past_img)
            else:
                trans_img = t(img)
            # end if

            # Save it
            past_img = trans_img

            # Paste to final image
            if new_img is None:
                new_img = trans_img
            else:
                # Both sizes
                tw, th = trans_img.size
                fw, fh = new_img.size

                # Check width
                if tw != fw:
                    raise Exception("Width of both image does not match {} != {}".format(tw, fw))
                # end if

                # Check mode
                if new_img.mode != trans_img.mode:
                    raise Exception("Images do not have the same mode {} != {}".format(new_img.mode, trans_img.mode))
                # end if

                # New image
                tmp_img = Image.new(new_img.mode, (tw, th + fh))

                # Paste
                tmp_img.paste(new_img, (0, 0))
                tmp_img.paste(trans_img, (0, fh))
                new_img = tmp_img
            # end if
        # end for

        return new_img
    # end __call__

    # Representation
    def __repr__(self):
        """
        Representation
        :return:
        """
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    # end __repr__

# end Concat
