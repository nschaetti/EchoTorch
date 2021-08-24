# -*- coding: utf-8 -*-
#
# File : echotorch/acf.py
# Description : Auto-covariance/correlation coefficients operations on (Time/Data/*)Tensor-
# Date : 24th of August, 2021
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
from typing import Dict, List, Callable
import matplotlib.pyplot as plt
import echotorch.viz

# Local imports
from .timetensors import TimeTensor
from .stat_ops import cov
from .base_ops import zeros


# Autocovariance coefficients for a time series (timetensor)
def autocovariance_coeffs(
        input: TimeTensor,
        k: int
) -> TimeTensor:
    r"""Returns the auto-covariance coefficients of a time series as a timetensor.

    :param input: input 0-D :math:`(T)` or 1-D :math:`(T, p)` time series.
    :type input: ``TimeTensor``
    :param k: number of different lags.
    :type k: ``int``
    :return: The auto-covariance coefficients as timetensor of size :math:`(k, p)` or :math:`(k)`.

    Example:

        >>> x = echotorch.rand(5, time_length=100)
        >>> autocov_coefs = echotorch.autocovariance_coeffs(x, k=50)
        >>> plt.figure()
        >>> echotorch.viz.timeplot(autocov_coeffs, title="Auto-covariance coefficients")
        >>> plt.show()
    """
    # Check k
    assert k > 1, "The number of lags must be greated than 1 (here {})".format(k)

    # Time dim must be first
    assert input.time_dim == 0, "Time dimension must be the first dimension of " \
                                "the timetensor (here {})".format(input.time_dim)

    # 0-D of 1-D time series
    assert input.cdim in [0, 1], "Expected 0-D or 1-D time series, found {}-D".format(input.cdim)

    # Difference with time length
    com_time_length = input.tlen - k

    # The time length for comparison must
    # be superior (or equal) to the number of lags required
    assert com_time_length >= k,  "Time length for comparison must be superior (or equal) to the number of lags " \
                                 "required (series of length {}, {} lags, " \
                                 "comparison length of {})".format(input.tlen, k, com_time_length)

    # Compute auto-covariance coefficients
    def compute_autocov_coeffs(x, lags):
        # Coeffs
        coeffs = zeros(length=lags + 1)

        # Variance
        coeffs[0] = cov(x[:com_time_length], x[:com_time_length])

        # For each lag k
        for lag_i in range(1, lags + 1):
            coeffs[lag_i] = cov(
                x[:com_time_length],
                x[lag_i:lag_i + com_time_length]
            )
        # end for

        return coeffs
    # end compute_autocov_coeffs

    # Compute coeffs for each channel
    if input.cdim == 0:
        return compute_autocov_coeffs(input, k)
    else:
        # Store coefs
        autocov_coefs = zeros(input.csize()[0], length=k + 1)

        # Compute coeffs for each channel
        for chan_i in range(input.csize()[0]):
            autocov_coefs[:, chan_i] = compute_autocov_coeffs(input[:, chan_i], k)
        # end for

        return autocov_coefs
    # end if
# end autocovariance_coeffs


# Auto-correlation coefficients for a time series (timetensor)
def acf(
        input: TimeTensor,
        k: int,
        coeffs_type: str = "covariance"
) -> TimeTensor:
    r"""Returns auto-correlation coefficients for a time series as a :class:`TimeTensor`.

    :param input: input 0-D :math:`(T)` or 1-D :math:`(T, p)` time series.
    :type input: ``TimeTensor``
    :param k: number of different lags.
    :type k: ``int``
    :return: The auto-covariance coefficients as timetensor of size :math:`(k, p)` or :math:`(k)`.
    :param coeffs_type: Type of coefficient, "covariance" or "correlation".
    :type coeffs_type: ``str``

    Example:

        >>> x = echotorch.rand(5, time_length=100)
        >>> autocor_coefs = echotorch.autocorrelation_coeffs(x, k=50)
        >>> plt.figure()
        >>> echotorch.viz.timeplot(autocor_coefs, title="Auto-correlation coefficients")
        >>> plt.show()
    """
    # Check type
    assert coeffs_type in ["covariance", "correlation"], "Unknown type of coefficients given, should be 'covariance' " \
                                                         "or 'correlation' ({} given)".format(coeffs_type)

    # Compute auto-covariance coefficients
    autocov_coeffs = autocovariance_coeffs(input, k)

    # Covariance
    if coeffs_type == "covariance":
        return autocov_coeffs
    elif coeffs_type == "correlation":
        # Normalize
        if autocov_coeffs.cdim == 0:
            return autocov_coeffs / autocov_coeffs[0]
        else:
            # For each channel
            for chan_i in range(autocov_coeffs.csize()[0]):
                autocov_coeffs[:, chan_i] /= autocov_coeffs[0, chan_i]
            # end for

            return autocov_coeffs
        # end if
    # end if
# end acf


# Cross-autocovariance Coefficients
def ccf(
        x: TimeTensor,
        y: TimeTensor,
        k: int,
        coeffs_type: str = "covariance"
) -> TimeTensor:
    r"""Returns cross auto-correlation coefficients (CCF) for two 0-D timeseries as a :class:`TimeTensor`.

    :param x:
    :type x:
    :param y:
    :type y:
    :param k:
    :type k:
    :param coeffs_type:
    :type coeffs_type:

    Example:

        >>>
    """
    # Check k
    assert k > 1, "The number of lags must be greated than 1 (here {})".format(k)

    # Check type
    assert coeffs_type in ["covariance", "correlation"], "Unknown type of coefficients given, should be 'covariance' " \
                                                         "or 'correlation' ({} given)".format(coeffs_type)

    # Check time series lengths
    assert x.tlen == y.tlen, "Expected two timeseries with same length (here {} != {})".format(x.tlen, y.tlen)

    # Same dim same size
    assert x.time_dim == y.time_dim, ""
    assert x.cdim == y.cdim, ""
    assert x.bdim == y.bdim, ""

    # Difference with time length
    com_time_length = x.tlen - k

    # The time length for comparison must
    # be superior (or equal) to the number of lags required
    assert com_time_length >= k, "Time length for comparison must be superior (or equal) to the number of lags " \
                                 "required (series of length {}, {} lags, " \
                                 "comparison length of {})".format(x.tlen, k, com_time_length)

    # Compute auto-covariance coefficients
    def compute_cross_autocov_coeffs(x, y, lags):
        # Coeffs
        coeffs = zeros(length=lags + 1)

        # Covariance
        coeffs[0] = cov(x[:com_time_length], y[:com_time_length])

        # For each lag k
        for lag_i in range(1, lags + 1):
            coeffs[lag_i] = cov(
                x[:com_time_length],
                y[lag_i:lag_i + com_time_length]
            )
        # end for

        return coeffs
    # end compute_autocov_coeffs

    # Compute auto-covariance coefficients
    autocov_coeffs = compute_cross_autocov_coeffs(x, y, k)

    # Covariance
    if coeffs_type == "covariance":
        return autocov_coeffs
    elif coeffs_type == "correlation":
        return autocov_coeffs / autocov_coeffs[0]
    # end if
# end ccf


# Cross-correlation


# Plot auto-covariance/correlation coefficients
def acfplot(
        input: TimeTensor,
        k: int,
        coeffs_type: str = "covariance",
        labels: List[str] = None,
        figure_params: Dict = None,
        plot_params: Dict = None
) -> None:
    r"""Plot auto-covariance or auto-correlation coefficients for a :class:`TimeTensor`.

    :param input: the input timetensor.
    :type input: :class:`TimeTensor`
    :param k: the number of lags.
    :type k: ``int``
    :param coeffs_type:
    :type coeffs_type:
    :param labels:
    :type labels:
    :param figure_params:
    :type figure_params:
    :param plot_params:
    :type plot_params:

    Example:

        >>> ...
    """
    # Only 0-D or 1-D timeseries
    assert input.cdim in [0, 1], "Expected 0-D or 1-D timeseries but {}-D given".format(input.cdim)

    # Compute coefficients
    acf_coeffs = acf(input, k, coeffs_type)

    # Labels
    if labels is None:
        labels = ["Series {}".format(i) for i in range(1, max(1, input.numelc() + 1))]
    else:
        assert len(labels) == input.numelc(), "The number of labels should be equal to the number of channels " \
                                                   "(here {} label given, " \
                                                   "{} channels)".format(len(labels), input.numelc())
    # end if

    # Init. params
    figure_params = {} if figure_params is None else figure_params
    plot_params = {} if plot_params is None else plot_params

    # Figure
    plt.figure(**figure_params)

    # Plot
    if input.cdim == 0:
        echotorch.viz.timeplot(acf_coeffs, label=labels[0], **plot_params)
    else:
        # For each channel
        for chan_i in range(input.numelc()):
            echotorch.viz.timeplot(acf_coeffs, label=labels[chan_i], **plot_params)
        # end for
    # end if

    # Show
    plt.show()
# end acfplot


# Alias for acfplot
def correlogram(
        input: TimeTensor,
        k: int,
        coeffs_type: str = "covariance",
        labels: List[str] = None,
        figure_params: Dict = None,
        plot_params: Dict = None
) -> None:
    r"""Alias for :func:`acfplot`
    """
    acfplot(input, k, coeffs_type, labels, figure_params, plot_params)
# end correlogram


# Plot cross auto-correlation coefficients
def ccfplot(
        x: TimeTensor,
        y: TimeTensor,
        k: int,
        coeffs_type: str = "covariance",
        labels: List[str] = None,
        figure_params: Dict = None,
        plot_params: Dict = None,
) -> None:
    r"""Plot cross auto-covariance or cross auto-correlation coefficients of two time series for a :class:`TimeTensor`.

    :param x: the input timetensor.
    :type x: :class:`TimeTensor`
    :param y: the input timetensor.
    :type y: :class:`TimeTensor`
    :param k: the number of lags.
    :type k: ``int``
    :param coeffs_type:
    :type coeffs_type:
    :param labels:
    :type labels:
    :param figure_params:
    :type figure_params:
    :param plot_params:
    :type plot_params:

    Example:

        >>> ...
    """
    # Only 0-D or 1-D timeseries
    assert x.cdim in [0, 1], "Expected 0-D or 1-D timeseries but {}-D given".format(x.cdim)
    assert y.cdim in [0, 1], "Expected 0-D or 1-D timeseries but {}-D given".format(y.cdim)

    # Compute coefficients
    ccf_coeffs = ccf(x, y, k, coeffs_type)

    # Labels
    if labels is None:
        labels = ["Series {}".format(i) for i in range(1, max(1, x.numelc() + 1))]
    else:
        assert len(labels) == x.numelc(), "The number of labels should be equal to the number of channels " \
                                                   "(here {} label given, " \
                                                   "{} channels)".format(len(labels), x.numelc())
    # end if

    # Init. params
    figure_params = {} if figure_params is None else figure_params
    plot_params = {} if plot_params is None else plot_params

    # Figure
    plt.figure(**figure_params)

    # Plot
    if input.cdim == 0:
        echotorch.viz.timeplot(ccf_coeffs, label=labels[0], **plot_params)
    else:
        # For each channel
        for chan_i in range(x.numelc()):
            echotorch.viz.timeplot(ccf_coeffs, label=labels[chan_i], **plot_params)
        # end for
    # end if

    # Show
    plt.show()
# end ccfplot


# Alias for acfplot
def cross_correlogram(
        x: TimeTensor,
        y: TimeTensor,
        k: int,
        coeffs_type: str = "covariance",
        labels: List[str] = None,
        figure_params: Dict = None,
        plot_params: Dict = None,
) -> None:
    r"""Alias for :func:`ccfplot`
    """
    ccfplot(x, y, k, coeffs_type, labels, figure_params, plot_params)
# end cross_correlogram


# Array of cross-autocorrelation
def ccfpairs(
        input: TimeTensor,
        k: int,
        coeffs_type: str = "covariance",
        labels: List[str] = None,
        figure_params: Dict = None,
        plot_params: Dict = None
) -> None:
    r"""Plot an array of plots with cross-autocorrelation for each pair of channel.

    :param input: the input timetensor.
    :type input: :class:`TimeTensor`
    :param k: the number of lags.
    :type k: ``int``
    :param coeffs_type:
    :type coeffs_type:
    :param labels:
    :type labels:
    :param figure_params:
    :type figure_params:
    :param plot_params:
    :type plot_params:

    Example:

        >>> ...
    """
    pass
# end ccfpairs

