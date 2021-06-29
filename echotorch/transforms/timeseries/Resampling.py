# -*- coding: utf-8 -*-
#
# File : echotorch/transforms/timeseries/Resampling.py
# Description : Resample a timeseries
# Date : 5th of February, 2021
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
# Copyright Nils Schaetti <nils.schaetti@unine.ch>, <nils.schaetti@unige.ch>


# Imports
import math
import torch
import torch.nn.functional
import torch.nn as nn
import numpy as np
from scipy import interpolate
from ..Transformer import Transformer


# Resampling transformer
class Resampling(Transformer):
    """
    Resample a timeseries
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, input_dim, scaling_factor=1.0, mode='scipy', kind='zero', dtype=torch.float64):
        """
        Constructor
        :param input_dim: Input dimension
        :param scaling_factor: Increasing/reducing factor for timeseries (1.0 returns the inputs)
        :param mode: Use 'torch' or 'scipy' for different interpolation method.
        """
        # Super constructor
        super(Resampling, self).__init__(input_dim=input_dim, output_dim=input_dim, dtype=dtype)

        # Properties
        self._scaling_factor = scaling_factor
        self._mode = mode
        self._kind = kind
    # end __init__

    # endregion CONSTRUCTORS

    # region PROPERTIES

    # Resampling factors
    @property
    def scaling_factor(self):
        """
        Scales
        """
        return self._scaling_factor
    # end scaling_factor

    # endregion PROPERTIES

    # region OVERRIDE

    # Transform
    def _transform(self, x):
        """
        Transform input
        :param x:
        :return:
        """
        if self._mode == 'torch':
            return torch.transpose(
                torch.squeeze(
                    torch.nn.functional.interpolate(
                        torch.unsqueeze(
                            torch.transpose(x, 0, 1),
                            dim=0
                        ),
                        scale_factor=(self._scaling_factor,),
                        recompute_scale_factor=True
                    )
                ),
                0,
                1
            )
        elif self._mode == 'scipy':
            # Sizes
            time_length, n_channels = x.shape

            # Reduced time length
            r_time_len = int(math.floor(time_length * self._scaling_factor))

            # Output tensor
            output = torch.zeros(r_time_len, n_channels)

            # For each channel
            for chan_i in range(n_channels):
                # Compute interpolation function
                chan_func = interpolate.interp1d(np.arange(0, time_length), x[:, chan_i].numpy(), kind=self._kind)

                # Downsampled data
                down_data = chan_func(np.linspace(0, time_length - 1, r_time_len))

                # Set output
                output[:, chan_i] = torch.from_numpy(down_data)
            # end for

            return output
        # end if
    # end _transform

    # endregion OVERRIDE

# end Resampling
