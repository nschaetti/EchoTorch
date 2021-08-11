# -*- coding: utf-8 -*-
#
# File : datasets/FuncDataset.py
# Description : Generic dataset to transform a function into a PyTorch Dataset object.
# Date : 9th of August, 2021
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
from typing import Tuple, List, Callable
import torch
import csv

# Local imports
from .EchoDataset import EchoDataset


# Generic dataset to transform a function into a PyTorch Dataset object.
class FuncDataset(EchoDataset):
    r"""Generic dataset to transform a function into a PyTorch Dataset object.
    """

    # Constructor
    def __init__(
            self,
            n: int,
            stream: bool,
            data_func: Callable,
            *args,
            **kwargs
    ) -> None:
        """
        Constructor

        Args:
            n: The Size of the dataset (the number of samples)
            stream: Do we generate samples on the fly?
            data_func: The callable object which will create the data.
            *args: Position arguments for the data function.
            **kwargs: Key arguments for the data function.
        """
        # Super
        super(FuncDataset, self).__init__(n, stream)

        # Properties
        self._data_func = data_func

        # Generate all samples if not streaming
        self._data = self._generate_dataset() if not stream else None
    # end __init__

    # region PRIVATE

    # Create the dataset by generating all samples
    def _generate_dataset(self) -> List[]:
        r"""Create the dataset by generating all samples (not streaming)

        Returns: Dataset samples as a list of timetensors

        """

    # endregion PRIVATE

    # region OVERRIDE

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx: Sample index
        :return: Sample as torch tensor
        """
        # Generate a Markov chain with
        # specified length.
        return self._data
    # end __getitem__

    # endregion OVERRIDE

    # region STATIC

    # Generate data
    def datafunc(self, *args, **kwargs) -> Tuple[torch.Tensor, List]:
        r"""Generate samples from the data function.

        Args:
            *args:
            **kwargs:

        Returns:

        """
        pass
    # end datafunc

    # endregion STATIC

# end DiscreteMarkovChainDataset
