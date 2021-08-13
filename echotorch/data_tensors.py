# -*- coding: utf-8 -*-
#
# File : echotorch/data_tensor.py
# Description : A special tensor with a key-to-index information
# Date : 13th of August, 2021
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
from typing import Optional, Tuple, Union, List, Callable, Any, Dict
import torch
import numpy as np

# Local imports
from .base_tensors import BaseTensor


# region DataIndexer
class DataIndexer(object):
    r"""Make the one-one relation between keys and indices
    """

    # Constructor
    def __init__(self, keys: List[Any]) -> None:
        r"""Create a data indexer from a dictionary

        :param keys:
        :type keys:
        """
        # Check keys (not ints)
        if not self._check_keys(keys):
            raise ValueError("Int cannot be used as key")
        # end if

        # Check if there is duplicates
        if len(keys) != len(set(keys)):
            raise ValueError("Key indexing of a tensor cannot accept duplicates")
        # end if

        # Properties
        self._size = len(keys)
        self._keys_to_indices = self._create_keys_to_indices(keys)
        self._indices_to_keys = self._create_indices_to_keys(keys)
    # end __init__

    # region PROPERTIES

    # Keys
    @property
    def keys(self) -> List[Any]:
        r"""List of keys
        """
        return list(self._keys_to_indices.keys())
    # end keys

    # Indices
    @property
    def indices(self) -> List[int]:
        r"""List of indices
        """
        return list(self._indices_to_keys.keys())
    # endregion PROPERTIES

    # region PUBLIC

    # To index
    def to_index(self, key: Union[List[Any], slice, Any]) -> Union[List[int], int, Dict, slice]:
        r"""From key to index
        """
        if isinstance(key, list):
            return [self.to_index(el) for el in key]
        elif isinstance(key, slice):
            if key.step is None:
                return slice(
                    self.to_index(key.start),
                    self.to_index(key.stop)
                )
            else:
                return slice(
                    self.to_index(key.start),
                    self.to_index(key.stop),
                    key.step
                )
            # end if
        elif isinstance(key, dict):
            return {k: self.to_index(v) for k, v in key.items()}
        else:
            if type(key) is int:
                return key
            else:
                return self._keys_to_indices[key]
            # end if
        # end if
    # end to_index

    # To keys
    def to_keys(self, index: Union[List[int], int, Dict[Any, int]]) -> Union[List[Any], Any]:
        r"""From index to keys
        """
        if isinstance(index, list):
            return [self.to_keys(el) for el in index]
        elif isinstance(index, dict):
            return {k: self.to_keys(v) for k, v in index.items()}
        else:
            return self._indices_to_keys[index]
        # end if
    # end to_keys

    # Slice keys
    def slice_keys(self, slice_item):
        r"""Slice keys
        """
        # Indices
        if slice_item.step is None:
            return self.to_keys(
                list(
                    range(
                        self.to_index(slice.start),
                        self.to_index(slice.stop)
                    )
                )
            )
        else:
            return self.to_keys(
                list(
                    range(
                        self.to_index(slice.start),
                        self.to_index(slice.stop),
                        slice.step
                    )
                )
            )
        # end if
    # end slice_keys

    # Filter items
    def filter_items(self, item) -> 'DataIndexer':
        r"""Create a new indexer with item filtered.
        """
        if isinstance(item, list):
            return DataIndexer(item)
        elif isinstance(item, slice):
            return DataIndexer(self.slice_keys(item))
        else:
            # Get index
            return DataIndexer([item])
        # end if
    # end filter_items

    # endregion PUBLIC

    # region PRIVATE

    # Check keys
    def _check_keys(self, keys: List[Any]) -> bool:
        r"""Check that there is not ints as keys
        """
        for key in keys:
            if type(key) is int:
                return False
            # end if
        # end for
        return True
    # end _check_keys

    # Compute keys to indices dictionary
    def _create_keys_to_indices(self, links: List[Any]) -> Dict[Any, int]:
        r"""Compute keys to indices dictionary
        """
        # Return key:index
        return {links[idx]: idx for idx in range(self._size)}
    # end _keys_to_indices

    # Compute indices to keys dictionary
    def _create_indices_to_keys(self, links: List[Any]) -> Dict[int, Any]:
        r"""Compute indices to keys dictionary
        """
        # Return index:key
        return {idx: links[idx] for idx in range(self._size)}
    # end _create_indices_to_keys

    # endregion PRIVATE

    # region OVERRIDE

    # Contains
    def __contains__(self, item):
        r"""Contains key
        """
        return item in self._keys_to_indices.keys
    # end __contains__

    # Get representation
    def __repr__(self) -> str:
        """
        Get a string representation
        """
        return "dataindexer(key_to_idx: {}, size:{})".format(
            self._keys_to_indices,
            self._size
        )
    # end __repr__

    # endregion OVERRIDE

# endregion DataIndexer


# Data Tensor
class DataTensor(BaseTensor):
    r"""A special tensor with a key-to-index information.
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'DataTensor'],
            keys: Optional[List[Union[DataIndexer, None]]] = None
    ) -> None:
        r"""DataTensor constructor

        :param data: The data in a torch tensor to transform to datatensor.
        :type data: ``torch.Tensor`` or ``DataTensor``
        :param keys: A dictionary with the target dimension index and a dictionary of key-to-index.
        :type keys: ``dict`` of ``dict``

        """
        # Base tensor
        super(DataTensor, self).__init__(data)

        # Init if None
        keys = [None] * data.ndim if keys is None else keys

        # We check that all keys corresponds to a dimension
        if len(keys) != data.ndim:
            raise ValueError(
                "Keys should be given for each dimension ({} sets of key for {} dims)".format(len(keys), data.ndim)
            )
        # end if

        # Set tensor and keys
        self._keys = self._build_keys(keys)
    # end __init__

    # endregion CONSTRUCTORS

    # region PROPERTIES

    # Get keys
    @property
    def keys(self) -> List[Union[None, DataIndexer]]:
        r"""Get keys
        """
        return self._keys
    # end keys

    # endregion PROPERTIES

    # region PUBLIC

    # Get the set of keys for a dimension
    def key_set(self, dim: int) -> List:
        r"""Get the set of keys for a dimension
        """
        # Get dict for this dim
        dim_indexer = self._keys[dim]

        # If not empty
        if dim_indexer is not None:
            return dim_indexer.keys()
        else:
            return list()
        # end if
    # end key_set

    # Get index for a key
    def get_index(self, dim: int, key: Any) -> Any:
        r"""Get index
        """
        if type(key) is int:
            return key
        else:
            # Get dict for this dim
            dim_indexer = self._keys[dim]

            # If not empty
            if dim_indexer is not None:
                # Return value
                return dim_indexer.to_index(key)
            else:
                return key
            # end if
        # end if
    # end get_index

    # Get key for an index
    def get_key(self, dim: int, key: Any) -> Any:
        r"""Get key for an index
        """
        if type(key) is int:
            # Get dict for this dim
            dim_indexer = self._keys[dim]

            # If not empty
            if dim_indexer is not None:
                # Return value
                return dim_indexer.to_keys(key)
            else:
                return None
            # end if
        else:
            return key
        # end if
    # end get_key

    # Key exists
    def is_in(self, dim: int, key: Any) -> bool:
        r"""Does the key exists.
        """
        # Get dict for this dim
        dim_indexer = self._keys[dim]

        # If not empty
        if dim_indexer is not None:
            # Return value
            return key in dim_indexer
        else:
            return False
        # end if
    # end is_in

    # endregion PUBLIC

    # region PRIVATE

    # Build data indexer
    def _build_keys(self, keys: List[Any]):
        r"""Build data indexers
        """
        key_indexers = list()
        for key in keys:
            if key is not None:
                key_indexers.append(DataIndexer(key))
            else:
                key_indexers.append(None)
            # end if
        # end for
        return key_indexers
    # end _build_keys

    # Transform indexing item
    def _trans_idx_item(self, item):
        r"""Transform indexing item
        """
        # List, tuple or list
        if isinstance(item, list):
            return [self.get_index(0, el) for el in item]
        elif isinstance(item, tuple):
            return tuple([self.get_index(el_i, el) for el_i, el in enumerate(item)])
        else:
            return self.get_index(0, item)
        # end if
    # end _trans_idx_item

    # Transform indexing item to keys
    def _trans_key_item(self, item):
        r"""Transform indexing item to keys
        """
        # List, tuple or list
        if isinstance(item, list):
            return [self.get_key(0, el) for el in item]
        elif isinstance(item, tuple):
            return tuple([self.get_key(el_i, el) for el_i, el in enumerate(item)])
        else:
            return self.get_key(0, item)
    # end _trans_key_item

    # endregion PRIVATE

    # region OVERRIDE

    # To
    def to(self, *args, **kwargs) -> 'DataTensor':
        r"""Performs TimeTensor dtype and/or device concersion. A ``torch.dtype`` and ``torch.device`` are inferred
        from the arguments of ``self.to(*args, **kwargs)

        .. note::
            From PyTorch documentation: if the ``self`` TimeTensor already has the correct ``torch.dtype`` and
            ``torch.device``, then ``self`` is returned. Otherwise, the returned timetensor is a copy of ``self``
            with the desired ``torch.dtype`` and ``torch.device``.

        Args:
            *args:
            **kwargs:

        Example::
            >>> ttensor = torch.randn(4, 4)
            >>> btensor = echotorch.datatensor(ttensor)
            >>> btensor.to(torch.float64)

        """
        # New tensor
        ntensor = self._tensor.to(*args, **kwargs)

        # Same tensor?
        if self._tensor == ntensor:
            return self
        else:
            return DataTensor(
                ntensor,
                keys=self._keys
            )
        # end if
    # end to

    # Get item
    def __getitem__(self, item) -> 'DataTensor':
        """
        Get data in the tensor
        """
        # Get data
        tensor_data = self._tensor[self._trans_idx_item(item)]

        # Get keys
        tensor_keys = self._trans_key_item(item)
        print("")
        print("item: {}".format(item))
        print("data: {}".format(tensor_data.size()))
        print("keys: {}".format(tensor_keys))
        return DataTensor(tensor_data, tensor_keys)
    # end __getitem__

    # Set item
    def __setitem__(self, key, value) -> None:
        """
        Set data in the tensor
        """
        self._tensor[key] = value
    # end __setitem__

    # Get representation
    def __repr__(self) -> str:
        """
        Get a string representation
        """
        return "datatensor({}, keys: {})".format(self._tensor, self._keys)
    # end __repr__

    # endregion OVERRIDE

# endregion DATATENSOR

