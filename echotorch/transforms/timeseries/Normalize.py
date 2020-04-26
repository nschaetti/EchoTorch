# -*- coding: utf-8 -*-
#
# File : echotorch/transforms/timeseries/Normalize.py
# Description : Normalize a timeserie.
# Date : 12th of April, 2020
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


# Normalize a timeseries
class Normalize(Transformer):
    """
    Normalize a timeserie
    """

    # Constructor
    def __init__(self, input_dim, mu, std, dtype=torch.float64):
        """
        Constructor
        """
        # Super constructor
        super(Normalize, self).__init__(
            input_dim=input_dim,
            output_dim=input_dim
        )

        # Properties
        self._mu = mu
        self._std = std
        self._input_dim = input_dim
        self._dtype = dtype
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

    # Output type
    @property
    def dtype(self):
        """
        Output type
        :return: Output type
        """
        return self._dtype
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
        return (x - self._mu) / self._std
    # end _transform

    #endregion PRIVATE

    #region OVERRIDE

    #endregion OVERRIDE

    #region STATIC

    #endregion STATIC

# end Normalize
