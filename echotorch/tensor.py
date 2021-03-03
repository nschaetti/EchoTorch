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
class TimeTensor(object):
    """
    A special tensor with a time dimension
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, time_length, size, time_dim=0, with_batch=False) -> None:
        """
        Constructor
        """
        # Properties
        self._time_length = time_length
        self._size = size
        self._time_dim = time_dim
        self._with_batch = with_batch
    # end __init__

    # endregion CONSTRUCTORS

    # region PROPERTIES

    # Time dimension
    @property
    def time_dim(self) -> int:
        """
        Time dimension
        """
        return self._time_dim
    # end time_time

    # Set time dimension
    @time_dim.setter
    def time_dim(self, value):
        """
        Set time dimension
        """
        self._time_dim = value
    # end time_dim

    # Has batch dimension?
    @property
    def with_batch(self) -> int:
        """
        Has batch dimension?
        """
        return self._with_batch
    # end with_batch

    # Has batch dimension?
    @with_batch.setter
    def with_batch(self, value):
        """
        Has batch dimension?
        """
        self._with_batch = value
    # end with_batch

    # endregion PROPERTIES

    # region PUBLIC

    # Add batch dim
    def add_batch(self):
        """
        Add batch dim
        """
        if not self._with_batch:
            uns_tensor = torch.unsqueeze(self, dim=0)
            uns_tensor.time_dim = self.time_dim + 1
            uns_tensor.with_batch = True
            return uns_tensor
        else:
            return self
        # end if
    # end add_batch

    # Length of timeseries
    def len(self) -> int:
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

    # region OVERRIDE

    # Deep Copy
    def __deepcopy__(self, memo):
        """
        Deep copy a time tensor
        """
        # Copy from tensor class
        tensor_copy = super(TimeTensor, self).__deepcopy__(memo)

        # Set time dimension
        tensor_copy.time_dim = self.time_dim
        tensor_copy.with_batch = self.with_batch
    # end __deepcopy__

    # Convert to double
    def double(self, memory_format=torch.preserve_format):
        """
        Convert to double
        """
        convert_tensor = self.to(torch.float64, memory_format=memory_format)
        # convert_tensor.time_dim = self.time_dim
        return convert_tensor
    # end double

    # Convert to long
    def long(self, memory_format=torch.preserve_format):
        """
        Convert to long
        """
        convert_tensor = self.to(torch.long, memory_format=memory_format)
        # convert_tensor.time_dim = self.time_dim
        return convert_tensor
    # end long

    # Convert to short
    def short(self, memory_format=torch.preserve_format):
        """
        Convert to short
        """
        convert_tensor = self.to(torch.short, memory_format=memory_format)
        # convert_tensor.time_dim = self.time_dim
        return convert_tensor
    # end short

    # To
    def to(self, *args, **kwargs):
        """
        To
        """
        new_tens = super(TimeTensor, self).to(*args, **kwargs)
        new_tens.time_dim = self._time_dim
        new_tens.with_batch = self._with_batch
        return new_tens
    # end to

    # Create new with same type and device
    def new_tensor(self, data, dtype=None, device=None, requires_grad=False, time_dim=0, with_batch=False) -> 'TimeTensor':
        """
        Create a new time tensor with same type and device
        """
        # New tensor
        new_tens = super(TimeTensor, self).new_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        new_tens.time_dim = time_dim
        new_tens.with_batch = with_batch
    # end new_tensor

    # Representation
    def __repr__(self):
        """
        Representation
        """
        # Prefix
        return 'timetensor(time_length={}, size={}, time_dim={}, with_batch={})'.format(
            self._time_length,
            self._size,
            self._time_dim,
            self._with_batch
        )
    # end __repr__

    # Tensor representation
    def tensor(self):
        """
        Tensor representation
        """
        return torch.empty(size=[self._time_length] + self._size)
    # end tensor

    # region STATIC

    # Create new tensor
    @staticmethod
    def __new__(cls, x, *args, time_dim=0, with_batch=False, **kwargs):
        """
        Create a new tensor
        :param time_dim:
        """
        print("__new__")
        # Create tensor
        ntensor = super().__new__(cls, x, *args, **kwargs)

        # New tensor size

        return ntensor
    # end __new__

    # endregion STATIC

# end TimeTensor
