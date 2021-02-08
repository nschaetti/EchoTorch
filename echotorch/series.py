# -*- coding: utf-8 -*-
#
# File : echotorch/series.py
# Description : Utility functions to generate timeseries
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
import echotorch.datasets


# Generate Copy Task series
def copytask(size, length_min, length_max, n_inputs, return_db=False, dtype=torch.float64):
    """
    Generate Copy Task series
    """
    if return_db:
        return echotorch.datasets.CopyTaskDataset(
            n_samples=size,
            length_min=length_min,
            length_max=length_max,
            n_inputs=n_inputs,
            dtype=dtype
        )
    else:
        return echotorch.datasets.CopyTaskDataset.generate(
            n_samples=size,
            length_min=length_min,
            length_max=length_max,
            n_inputs=n_inputs,
            dtype=dtype
        )
    # end if
# end copytask


# Generate Discrete Markov Chain dataset
def discrete_markov_chain(size, length, n_states, probability_matrix, start_state=0, return_db=False,
                          dtype=torch.float64):
    """
    Generate Discrete Markov Chain dataset
    :param size:
    :param length:
    :param n_states:
    :param probability_matrix:
    :param start_state:
    :param return_db:
    :param dtype:
    """
    if return_db:
        return echotorch.datasets.DiscreteMarkovChainDataset(
            n_samples=size,
            sample_length=length,
            probability_matrix=probability_matrix
        )
    else:
        samples = list()
        for sample_i in range(size):
            samples.append(echotorch.datasets.DiscreteMarkovChainDataset.generate(
                length=length,
                n_states=n_states,
                probability_matrix=probability_matrix,
                start_state=start_state,
                dtype=dtype
            ))
        # end for
        return samples
    # end if
# end discrete_markov_chain


# Load Time series from a CSV file
def csv_file(csv_file, delimiter, quotechar, columns, return_db=False, dtype=torch.float64):
    """
    Load Timeseries from a CSV file
    :param csv_file:
    :param delimiter:
    :param quotechar:
    :param columns:
    :param return_db:
    :param dtype:
    """
    if return_db:
        return echotorch.datasets.FromCSVDataset(
            csv_file=csv_file,
            columns=columns,
            delimiter=delimiter,
            quotechar=quotechar
        )
    else:
        pass
    # end if
# end csv_file
