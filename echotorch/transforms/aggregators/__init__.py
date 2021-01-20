# -*- coding: utf-8 -*-
#
# File : echotorch/transforms/aggregators/__init__.py
# Description : Aggregators subpackage init file.
# Date : 19th of January, 2021
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
from .ACFAggregator import ACFAggregator
from .ChainAggregator import ChainAggregator
from .CovMatrixAggregator import CovMatrixAggregator
from .SetAggregator import SetAggregator
from .StatsAggregator import StatsAggregator

# ALL
__all__ = [
    'ACFAggregator', 'ChainAggregator', 'CovMatrixAggregator', 'SetAggregator', 'StatsAggregator'
]