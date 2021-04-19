# -*- coding: utf-8 -*-
#
# File : echotorch/datasets/EchoDataset.py
# Description : Base class for EchoTorch datasets
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
from typing import Union, List
import torch
from torch.utils.data.dataset import Dataset

# Local imports
# from .DatasetComposer import DatasetComposer


# EchoDataset
class EchoDataset(Dataset):
    """
    Base class for EchoTorch datasets
    """

    # region OVERRIDE

    # Representation
    def __repr__(self):
        """
        Representation
        """
        return "{}({})".format(
            self.__class__.__name__,
            self.extra_repr()
        )
    # end __repr__

    # Override addition
    def __add__(self, other):
        """
        Override addition
        @param other: The other EchoDataset
        @return: self + other
        """
        """return DatasetComposer(
            [
                self,
                other
            ]
        )"""
        return None
    # end __add__

    # Override addition
    def __radd__(self, other):
        """
        Override addition
        @param other: The other EchoDataset
        @return: other + self
        """
        """return DatasetComposer(
            [
                self,
                other
            ]
        )"""
        return None
    # end __radd__

    # endregion OVERRIDE

    # region TO_IMPLEMENT

    # Get the whole dataset
    def generate_full(self) -> Union[torch.Tensor, List]:
        """
        Get the whole dataset (according to init parameters)
        @return: The Torch Tensor
        """
        raise Exception("generate_full not implemented")
    # end generate_full

    # Extra representation
    def extra_repr(self):
        """
        Extra representation
        """
        return ""
    # end extra_repr

    # Function to generate data
    @staticmethod
    def datafunc(*args, **kwargs):
        """
        Function to generate data
        :param args: Positional arguments
        :param kwargs: Arguments
        """
        pass
    # end datafunc

    # endregion TO_IMPLEMENT

# end EchoDataset
