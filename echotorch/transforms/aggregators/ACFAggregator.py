# -*- coding: utf-8 -*-
#
# File : echotorch/transforms/aggregators/ACFAggregator.py
# Description : Compute Auto-Covariance Coefficients on each channels.
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


# Auto-correlation Coefficients Function (ACF) aggregator
class ACFAggregator(Aggregator):
    """
    An aggregator which compute the Auto-correlation Coefficients Function (ACF)
    of input time series and compute the average.
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, n_lags, channels, **kwargs):
        """
        Constructor
        :param n_lags:
        :param kwargs:
        """
        # Properties
        self._n_lags = n_lags
        self._channels = channels
        self._n_channels = len(channels)

        # To Aggregator
        super(ACFAggregator, self).__init__(**kwargs)
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

    # Compute auto-covariance coefficients for a time series
    def _auto_cov_coefs(self, x):
        """
        Compute auto-covariance coefficients for a time series
        :param x: A (time length x n channels) data tensor
        :return: The auto-covariance coefficients for each lag
        """
        # Store coefs
        autocov_coefs = torch.zeros(self._n_lags)

        # For each lag
        for lag_i in range(self._n_lags):
            autocov_coefs[lag_i] = self._cov(x[:-lag_i], x[lag_i:])
        # end for

        # Normalize with first coef
        autocov_coefs /= autocov_coefs[0]

        return autocov_coefs
    # end _auto_cov_coefs

    # endregion PRIVATE

    # region OVERRIDE

    # Initialize
    def _initialize(self):
        """
        Initialize
        """
        # For each lag and channels
        for channel_i in range(self._n_channels):
            self._create_entry("autocov_coefs_" + str(channel_i), torch.zeros(self._n_lags))
        # end for
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
            raise Exception("Cannot compute ACF for a one time step time series")
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

            # For each channels
            for channel_i in range(n_channels):
                # Entry name
                entry_name = "autocov_coefs_" + str(channel_i)

                # Compute autocov coefs
                autocov_coefs = self._auto_cov_coefs(batch_data[:, channel_i])

                # Update entry
                self._update_entry(entry_name, self[entry_name] + autocov_coefs)
            # end for
        # end for
    # end _aggregate

    # Finalize
    def _finalize(self):
        """
        Finalize aggregation
        """
        # For each channel
        for channel_i in range(self._n_channels):
            entry_name = "autocov_coefs_" + str(channel_i)
            self._update_entry(entry_name, self[entry_name] / self.get_counter(entry_name), inc=False)
        # end for

        # Finalized
        self._finalized = True
    # endregion OVERRIDE

# end ACFAggregator
