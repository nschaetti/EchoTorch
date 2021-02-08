# -*- coding: utf-8 -*-
#
# File : echotorch/modules.py
# Description : Utility functions to create modules
# Date : 5th of February, 2021
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
import echotorch.nn as etnn
import echotorch.utils.matrix_generation as etmm


# Create an Echo State Network (ESN)
def esn(input_dim, hidden_dim, output_dim, spectral_radius, leaky_rate, w_connectivity,
        win_connectivity, wbias_connectivity, input_scaling, bias_scaling, ridge_param,
        softmax_output=False, dtype=torch.float64):
    """
    Create an Echo State Network (ESN)
    """
    # Internal matrix generator
    w_generator = etmm.matrix_factory.get_generator(
        "uniform",
        connectivity=w_connectivity,
        spectral_radius=spectral_radius
    )

    # Input-to-reservoir generator
    win_generator = etmm.matrix_factory.get_generator(
        "uniform",
        connectivity=1.0 if hidden_dim < 100 else win_connectivity,
        apply_spectral_radius=False,
        scale=input_scaling
    )

    # Reservoir bias generator
    wbias_generator = etmm.matrix_factory.get_generator(
        "uniform",
        connectivity=wbias_connectivity,
        apply_spectral_radius=False,
        scale=bias_scaling
    )

    # Create the ESN
    return etnn.LiESN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        leaky_rate=leaky_rate,
        w_generator=w_generator,
        win_generator=win_generator,
        wbias_generator=wbias_generator,
        input_scaling=input_scaling,
        ridge_param=ridge_param,
        softmax_output=softmax_output,
        dtype=dtype
    )
# end esn
