# -*- coding: utf-8 -*-
#
# File : echotorch/transforms/timeseries/FourierTransform.py
# Description : Perform Fourier transform on a timeserie
# Date : 19th of August, 2020
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
import torch
from ..Transformer import Transformer


# Perform Fourier transform on a timeserie
class FourierTransform(Transformer):
    """
    Perform Fourier transform on a timeserie
    """

    # Constructor
    def __init__(self, signal_dim, normalize=False):
        """
        Constructor
        """
        # Super constructor
        super(FourierTransform, self).__init__(
            input_dim=signal_dim,
            output_dim=signal_dim
        )

        # Properties
        self.normalize = normalize
    # end __init__

    #region PROPERTIES

    # Dimension of the input timeseries
    @property
    def input_dim(self):
        """
        Dimension of the output timeseries
        :return: Dimension of the output timeseries
        """
        return self._input_dim
    # end output_dim

    # Dimension of the output timeseries
    @property
    def output_dim(self):
        """
        Dimension of the output timeseries
        :return: Dimension of the output timeseries
        """
        return self._output_dim
    # end output_dim

    #endregion PROPERTIES

    #region PRIVATE

    # Transform
    def _transform(self, x):
        """
        Transform input
        :param x:
        :return:
        """
        return torch.fft(x, self.input_dim, self.normalize)
    # end _transform

    #endregion PRIVATE

    #region OVERRIDE

    #endregion OVERRIDE

    #region STATIC

    #endregion STATIC

# end Normalize
