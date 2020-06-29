# -*- coding: utf-8 -*-
#
# File : examples/conceptors/patterns/periodic_patterns.py
# Description : Set of periodic pattern for tests
# Date : 29th of June, 2020
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
import echotorch.datasets as etds
import torch


# Pattern settings
pattern_settings = [
    {'period': 8.8342522, 'type': 'sine'},
    {'period': 9.8342522, 'type': 'sine'},
    {'period': [-1, -1, 0.0, 0.25, 1.0], 'type': 'pattern'},
    {'period': [-1, -0.75, 0.0, 0.5, 1.0], 'type': 'pattern'},
    {
        'period': [0.9000000000000002, -0.11507714997817164, 0.17591170369788622, -0.9, -0.021065045054201592],
        'type': 'pattern'
    },
    {
        'period': [0.9, -0.021439412841318672, 0.0379515995051003, -0.9, 0.06663989939293802],
        'type': 'pattern'
    }
]

# Number of patterns
n_patterns = len(pattern_settings)


# Get periodic pattern from ID
def pattern_library(pattern_id, washout_length, learn_length, n_samples=1, dtype=torch.float32):
    """
    Get periodic patterm from ID
    :param pattern_id: Pattern's ID
    :param washout_length: Washout length
    :param learn_length: Learn length
    :param n_samples: Number of samples in the dataset
    :param dtype: Data type (default, float32)
    :return: PeriodicSignalDataset object
    """
    # Check pattern ID
    if pattern_id >= len(pattern_settings):
        raise Exception("Unknown pattern with ID : {}".format(pattern_id))
    # end if

    # Create dataset, return
    if pattern_settings[pattern_id]['type'] == 'pattern':
        return etds.PeriodicSignalDataset(sample_len=washout_length + learn_length, n_samples=1,
            period=pattern_settings[pattern_id]['period'],
            dtype=dtype
        )
    elif pattern_settings[pattern_id]['type'] == 'sine':
        return etds.SinusoidalTimeseries(
            sample_len=washout_length + learn_length,
            n_samples=1,
            a=1,
            period=pattern_settings[pattern_id]['period'],
            dtype=dtype
        )
    else:
        raise Exception("Unknown pattern type : {}".format(pattern_settings[pattern_id]['type']))
    # end if
# end pattern_library
