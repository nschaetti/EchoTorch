# -*- coding: utf-8 -*-
#
# File : echotorch/transforms/timeseries/AddNoise.py
# Description : Add noise to a timeserie
# Date : 21th of August, 2020
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


# AddNoise
class AddNoise(Transformer):
    """
    Add noise to a timeserie
    """

    # Constructor
    def __init__(self, input_dim, mu, std):
        """
        Constructor
        :param input_dim: Dimension of the timeseries
        :param mu: The tensor of per-element means of noise
        :param std: The tensor of per-element means of noise
        :param dtype: Data type
        """
        # Super constructor
        super(AddNoise, self).__init__(
            input_dim=input_dim,
            output_dim=input_dim
        )

        # Properties
        self._mu = mu
        self._std = std
        self._input_dim = input_dim
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
        :param x: The input timeserie (T x input_dim)
        :return: The timeseries with additional noise
        """
        # Add noise to each dimension
        for i in range(self._input_dim):
            noise = torch.normal(mean=self._mu[i], std=self._std[i], size=(1, x.size(0)))[0]
            x[:, i] += noise
        # end for

        return x
    # end _transform

    #endregion PRIVATE

    #region OVERRIDE

    #endregion OVERRIDE

    #region STATIC

    #endregion STATIC

# end AddNoise
