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
import sys

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
    def __init__(self, input_dim, n_lags, *args, **kwargs):
        """
        Constructor
        :param n_lags:
        :param kwargs:
        """
        # Properties
        self._n_lags = n_lags
        self._channels = [i for i in range(input_dim)]

        # To Aggregator
        super(ACFAggregator, self).__init__(input_dim, *args, **kwargs)
    # end __init__

    # endregion CONSTRUCTOR

    # region PUBLIC

    # Get auto-covariance coefficients
    def coefficients(self, entry_name):
        """
        Get auto-covariance coefficients
        """
        return self._data[entry_name] / self._counters[entry_name]
    # end coefficients

    # endregion PUBLIC

    # region PRIVATE

    # Compute covariance for a  lag
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
        :param x: A (time length) data tensor
        :return: The auto-covariance coefficients for each lag
        """
        # Store coefs
        autocov_coefs = torch.zeros(self._n_lags)

        # Time length for comparison
        com_time_length = x.size(0) - self._n_lags

        # Covariance t to t
        autocov_coefs[0] = self._cov(x[:-com_time_length], x[:-com_time_length])

        # For each lag
        for lag_i in range(1, self._n_lags):
            autocov_coefs[lag_i] = self._cov(
                x[:com_time_length],
                x[lag_i:lag_i+com_time_length]
            )
        # end for

        # Co
        c0 = autocov_coefs[0].item()

        # Normalize with first coef
        autocov_coefs /= c0

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
        for channel in self._channels:
            self._register("autocov_coefs_" + str(channel), torch.zeros(self._n_lags))
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
        if x.ndim == 1:
            raise Exception("Cannot compute ACF for a one time step time series")
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

        # Check the length of the TS
        if time_length >= self._n_lags + 1:
            # Compute covariance for each batch
            for batch_i in range(batch_size):
                # Batch data, we want time before
                # channels
                batch_data = x[batch_i] if time_dim == 1 else torch.transpose(x[batch_i], 0, 1)

                # For each channels
                for channel in self._channels:
                    # Entry name
                    entry_name = "autocov_coefs_" + str(channel)

                    # Compute autocov coefs
                    autocov_coefs = self._auto_cov_coefs(batch_data[:, channel])

                    # Update entry
                    self._data[entry_name] += autocov_coefs
                    self._inc(entry_name)
                # end for
            # end for
        # end if
    # end _aggregate

    # endregion OVERRIDE

# end ACFAggregator
