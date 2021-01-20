# -*- coding: utf-8 -*-
#
# File : echotorch/transforms/aggregators/CovMatrixAggregator.py
# Description : Compute covariance matrix between the different channels.
# Date : 18th of January, 2021
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

# EchoTorch imports
from echotorch.transforms import Aggregator


# Covariance matrix aggregator
class CovMatrixAggregator(Aggregator):
    """
    An aggregator which compute the covariance matrix between channels
    of input time series and compute the average.
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, input_dim, *args, **kwargs):
        """
        Constructor
        :param input_dim:
        :param args:
        :param kwargs:
        """
        # To Aggregator
        super(CovMatrixAggregator, self).__init__(input_dim, *args, **kwargs)
    # end __init__

    # endregion CONSTRUCTOR

    # region PUBLIC

    # Get covariance matrix
    def covariance_matrix(self):
        """
        Get covariance matrix
        """
        return self._data['cov_matrix'] / self._counters['cov_matrix']
    # end covariance_matrix

    # endregion PUBLIC

    # region PRIVATE

    # endregion PRIVATE

    # region OVERRIDE

    # Initialize
    def _initialize(self):
        """
        Initialize
        """
        self._register("cov_matrix", torch.zeros(self._input_dim, self._input_dim))
        self._initialized = True
    # end _initialize

    # Aggregate information
    def _aggregate(self, x):
        """
        Aggregate information
        :param x: Input tensor data
        """
        # Add batch if not present
        if x.ndim == 1:
            raise Exception("Cannot compute covariance for a one time step time series")
        elif x.ndim == 2:
            x = torch.unsqueeze(x, dim=0)
            time_dim = self._time_dim + 1
        else:
            time_dim = self._time_dim
        # end if

        # Sizes
        if time_dim == 1:
            batch_size, time_length, n_channels = x.size()
        else:
            batch_size, n_channels, time_length = x.size()
        # end if

        # Compute covariance for each batch
        for batch_i in range(batch_size):
            # Batch data
            batch_data = x[batch_i] if time_dim == 1 else torch.transpose(x[batch_i], 0, 1)

            # Covariance matrix
            cov_matrix = torch.matmul(batch_data.t(), batch_data) / time_length

            # Update entry
            self._data['cov_matrix'] += cov_matrix
            self._inc('cov_matrix')
        # end for
    # end _aggregate

    # endregion OVERRIDE

# end CovMatrixAggregator
