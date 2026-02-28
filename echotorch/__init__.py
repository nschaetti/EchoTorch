# -*- coding: utf-8 -*-
#
# File : echotorch/__init__.py
# Description : EchoTorch package init file.
# Date : 26th of January, 2018
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

# Imports
from . import datasets
from . import evaluation
from . import models
from . import nn
from . import transforms
from . import utils
from .timetensor import TimeTensor
from .base_ops import (
    timetensor, as_timetensor, sparse_coo_timetensor, as_strided, from_numpy,
    zeros, zeros_like, ones, ones_like, arange, linspace, logspace,
    empty, empty_like, empty_strided, full, full_like,
    quantize_per_timetensor, quantize_per_channel, dequantize, complex, polar,
    rand, randn, tcat, cat, tindex_select, is_timetensor,
)
from .stat_ops import tsum, tquantile, tmean, tstd, tvar, cor, cov


# All echotorch's modules
__all__ = [
    'datasets', 'evaluation', 'models', 'nn', 'transforms', 'utils',
    'TimeTensor', 'timetensor', 'as_timetensor', 'sparse_coo_timetensor', 'as_strided', 'from_numpy',
    'zeros', 'zeros_like', 'ones', 'ones_like', 'arange', 'linspace', 'logspace',
    'empty', 'empty_like', 'empty_strided', 'full', 'full_like',
    'quantize_per_timetensor', 'quantize_per_channel', 'dequantize', 'complex', 'polar',
    'rand', 'randn', 'tcat', 'cat', 'tindex_select', 'is_timetensor',
    'tsum', 'tquantile', 'tmean', 'tstd', 'tvar', 'cor', 'cov'
]
