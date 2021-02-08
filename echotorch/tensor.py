# -*- coding: utf-8 -*-
#
# File : echotorch/timetensor.py
# Description : A special tensor with a time dimension
# Date : 25th of January, 2021
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


# Import
import torch


# TimeTensor
class TimeTensor(torch.Tensor):
    """
    A special tensor with a time dimension
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, x, time_dim, with_batch=False):
        """
        Constructor
        """
        # Properties
        self._time_dim = time_dim
        self._with_batch = with_batch
    # end __init__

    # endregion CONSTRUCTORS

    # region PROPERTIES

    # Time dimension
    @property
    def time_dim(self):
        """
        Time dimension
        """
        return self._time_dim
    # end time_time

    # endregion PROPERTIES

    # region PUBLIC

    # Length of timeseries
    def len(self):
        """
        Length of timeseries
        """
        if self._with_batch:
            pass
        else:
            return self.size(self._time_dim)
        # end if
    # end lengths

    # endregion PUBLIC

    # region STATIC

    # Create new tensor
    @staticmethod
    def __new__(cls, x, time_dim, *args, with_batch=False, **kwargs):
        """
        Create a new tensor
        :param time_dim:
        """
        # Create tensor
        ntensor = super().__new__(cls, x, *args, **kwargs)

        # New tensor size

        return ntensor
    # end __new__

    # endregion STATIC

# end TimeTensor
