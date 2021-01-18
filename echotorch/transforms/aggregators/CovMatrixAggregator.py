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
    def __init__(self, channels, **kwargs):
        """
        Constructor
        :param n_lags:
        :param kwargs:
        """
        # Properties
        self._channels = channels
        self._n_channels = len(channels)

        # To Aggregator
        super(CovMatrixAggregator, self).__init__(**kwargs)
    # end __init__

    # endregion CONSTRUCTOR

    # region PRIVATE

    # Compute covariance for a lag
    def _cov(self, x, y):
        """
        Compute covariance for a lag
        :param x: t positioned time series (time length, n channels)
        :param y: t + k positioned time series (time length, n channels)
        :return: The covariance coefficients
        """
        # Average x
        x_mu = torch.mean(x, dim=0)

        # Average covariance over length
        return torch.mean(torch.mul(x - x_mu, y - x_mu))
    # end _cov

    # endregion PRIVATE

    # region OVERRIDE

    # Initialize
    def _initialize(self):
        """
        Initialize
        """
        self._create_entry("cov_matrix", torch.zeros(self._n_channels, self._n_channels))
        self._initialized = True
    # end _initialize

    # Aggregate information
    def _aggregate(self, x):
        """
        Aggregate information
        :param x: Input tensor data
        """
        # Add batch if not present
        if x.ndim() == 1:
            raise Exception("Cannot compute covariance for a one time step time series")
        elif x.ndim() == 2:
            x = torch.unsqueeze(x, dim=0)
            self._time_dim += 1
        # end if

        # Sizes
        if self._time_dim == 1:
            batch_size, time_length, n_channels = x.size()
        else:
            batch_size, n_channels, time_length = x.size()
        # end if

        # Compute covariance for each batch
        for batch_i in range(batch_size):
            # Batch data
            batch_data = x[batch_i] if self._time_dim == 1 else torch.transpose(x[batch_i], 0, 1)

            # Current covariance matrix
            cov_matrix = torch.zeros(n_channels, n_channels)

            # For each pair of channel
            for channel1_i in range(n_channels):
                for channel2_i in range(n_channels):
                    cov_matrix[channel1_i, channel2_i] = self._cov(
                        batch_data[:, channel1_i],
                        channel2_i[:, channel2_i]
                    )
                # end for
            # end for

            # Update entry
            self._update_entry('cov_matrix', self['cov_matrix'] + cov_matrix)
        # end for
    # end _aggregate

    # Finalize
    def _finalize(self):
        """
        Finalize aggregation
        """
        # Average
        self._update_entry('cov_matrix', self['cov_matrix'] / self.get_counter('cov_matrix'), inc=False)

        # For each
        self._finalized = True
    # endregion OVERRIDE

# end ACFAggregator
