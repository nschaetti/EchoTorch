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
import torch
from torch.utils.data.dataset import Dataset


# EchoDataset
class EchoDataset(Dataset):
    """
    Base class for EchoTorch datasets
    """

    # region TO_IMPLEMENT

    @staticmethod
    def datafunc(*args, **kwargs):
        """
        Function to generate data
        :param args: Positional arguments
        :param kwargs: Arguments
        """
        pass
    # end datafunc

# end EchoDataset
