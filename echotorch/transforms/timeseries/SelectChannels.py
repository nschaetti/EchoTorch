# -*- coding: utf-8 -*-
#
# File : echotorch/transforms/timeseries/SelectChannels.py
# Description : Select channels in a timeseries
# Date : 20th of January, 2021
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


# SelectChannels
class SelectChannels(Transformer):
    """
    Select channels
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, input_dim, channels, time_dim=0):
        """
        Constructor
        """
        # Super constructor
        super(SelectChannels, self).__init__(
            input_dim=input_dim,
            output_dim=len(channels)
        )

        # Properties
        self._channels = channels
        self._time_dim = time_dim
    # end __init__

    # endregion CONSTRUCTORS

    # region PROPERTIES

    # Channels
    @property
    def channels(self):
        """
        Channels
        """
        return self._channels
    # end channels

    # endregion PROPERTIES

    # region OVERRIDE

    # Transform
    def _transform(self, x):
        """
        Transform
        """
        if self._time_dim == 0:
            return x[:, self._channels]
        else:
            return x[self._channels, :]
        # end if
    # end _transform

    # endregion OVERRIDE

# end SelectChannels
