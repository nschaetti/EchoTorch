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

# Imports
import torch


# TimeTensor
class TimeTensor(object):
    """A  special tensor with a time and a batch dimension
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, data, time_dim=0, with_batch=False, **kwargs):
        # Transform to tensor
        if type(data) is TimeTensor:
            tensor_data = data.tensor
        else:
            tensor_data = torch.as_tensor(data, **kwargs)
        # end if

        # The tensor must have enough dimension
        # for the time dimension
        if tensor_data.ndim < time_dim + 1:
            # Error
            raise ValueError(
                "Time dimension does not exists in the data tensor "
                "(time dim at {}, {} dimension in tensor".format(time_dim, tensor_data.ndim)
            )
        # end if

        # If there is a batch dimension, time dimension cannot
        # be zero
        if with_batch and time_dim == 0:
            # Error
            raise ValueError("Time dimension cannot be the same as the batch dimension (batch is first)")
        # end if

        # Check dimension
        if with_batch and tensor_data.ndim <= 2:
            raise ValueError(
                "Time tensor with batch dimension must have "
                "at least two dimensions (here {}".format(tensor_data.ndim)
            )
        # end if

        # Properties
        self._tensor = tensor_data
        self._time_dim = time_dim
        self._with_batch = with_batch
    # end __init__

    # endregion CONSTRUCTORS

    # region PROPERTIES

    # Get tensor
    @property
    def tensor(self):
        """Get tensor
        """
        return self._tensor
    # end tensor

    # Set tensor
    @tensor.setter
    def tensor(self, value):
        """Set tensor
        """
        raise RuntimeError("You cannot set the tensor by yourself")
    # end tensor

    # Time dimension (getter)
    @property
    def time_dim(self):
        """Time dimension (getter)
        """
        return self._time_dim
    # end time_dim

    # Time dimension (setter)
    @time_dim.setter
    def time_dim(self, value):
        """Time dimension (setter)
        """
        # Check time and batch dim not overlapping
        if self.with_batch and value == 0:
            raise ValueError("Time dimension cannot be the same as the batch dimension (batch is first)")
        # end if

        # Check time dim is valid
        if value >= self.tensor.ndim:
            # Error
            raise ValueError(
                "Time dimension does not exists in the data tensor "
                "(new time dim at {}, {} dimension in tensor".format(value, self._tensor.ndim)
            )
        # end if

        # Set new time dim
        self._time_dim = value
    # end time_dim

    # With batch (getter)
    @property
    def with_batch(self):
        """With batch (getter)
        """
        return self._with_batch
    # end with_batch

    # Time length
    @property
    def tlen(self):
        """Time length
        """
        return self._tensor.size(self._time_dim)
    # end tlen

    # Batch size
    @property
    def batch_size(self):
        """Batch size
        """
        if self.with_batch:
            return self._tensor.size(0)
        else:
            return None
        # end if
    # end batch_size

    # Number of dimension
    @property
    def ndim(self):
        """Number of dimension
        """
        return self._tensor.ndim
    # end ndim

    # Data type
    @property
    def dtype(self):
        """Data type.
        """
        return self._tensor.dtype
    # end dtype

    # endregion PROPERTIES

    # region PUBLIC

    # Size
    def size(self):
        """Size
        """
        return self._tensor.size()
    # end size

    # Size of time-related dimension
    def tsize(self):
        """Size of time-related dimension
        """
        tensor_size = self._tensor.size()
        return tensor_size[self.time_dim+1:]
    # end tsize

    # Long
    def long(self):
        """Long.
        """
        self._tensor = self._tensor.long()
    # end long

    # endregion PUBLIC

    # region OVERRIDE

    # Get item
    def __getitem__(self, item):
        """Get item.
        """
        return self._tensor[item]
    # end __getitem__

    # Set item
    def __setitem__(self, key, value):
        """Set item."""
        self._tensor[key] = value
    # end __setitem__

    # Length
    def __len__(self):
        """Length
        """
        return self.tlen
    # end __len__

    # Get representation
    def __repr__(self):
        """Get representation
        """
        return "time dim:\n{}\n\nwith batch:\n{}\n\ndata:\n{}".format(self._time_dim, self._with_batch, self._tensor)
    # end __repr__

    # Torch functions
    def __torch_function__(self, func, types, args=(), kwargs=None):
        """Torch functions
        """
        # Dict if None
        if kwargs is None:
            kwargs = {}
        # end if

        # Convert timetensor to tensors
        def convert(args):
            if type(args) is TimeTensor:
                return args.tensor
            elif type(args) is tuple:
                return tuple([convert(a) for a in args])
            elif type(args) is list:
                return [convert(a) for a in args]
            else:
                return args
            # end if
        # end convert

        # Get the tensor in the arguments
        args = [convert(a) for a in args]

        # Execute function
        ret = func(*args, **kwargs)

        # Return a new time tensor
        return TimeTensor(ret, time_dim=self._time_dim, with_batch=self._with_batch)
    # end __torch_function__

    # endregion OVERRIDE

# end TimeTensor
