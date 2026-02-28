# -*- coding: utf-8 -*-
#
# File : echotorch/series_ops.py
# Description : Series transformation operations
# Date : 18th of August, 2021
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
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>
# University of Geneva <nils.schaetti@unige.ch>

# Imports
from .timetensors import TimeTensor


# Difference operator
def diff(
        input: TimeTensor
) -> TimeTensor:
    r"""The difference operator.

    The difference operator compute the difference between the time series :math:`x(t)` at time :math:`t` and :math:`t+1`.

    .. :math::
        diff(x) = x(t+1) - x(t)

    :param input: input timeseries.
    :type input: ``TimeTensor``
    :return: The difference between :math:`x` at time :math:`t` and :math:`t+1` as a ``TimeTensor``.

    Example:

        >>> x = echotorch.rand(5, time_length=100)
        >>> df = echotorch.diff(x)
    """
    # Time length must be > 1
    if input.tlen <= 1:
        raise ValueError("The input timeseries must have at least a length equal to 2 (here {})".format(input.tlen))
    # end if

    # Construct the indexer
    t_index = [slice(None, None)] * input.bdim
    t_index += [slice(None, -1)]

    # Indexers
    tp_index = list(t_index)
    tp_index[input.time_dim] = slice(1, None)

    return input[tuple(tp_index)] - input[tuple(t_index)]
# end diff

