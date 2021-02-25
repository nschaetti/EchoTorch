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


# Generate Copy Task seriesJFJJFJ
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
        return echotorch.datasets.FromCSVDataset.generate(
            csv_file=csv_file,
            delimiter=delimiter,
            quotechar=quotechar,
            columns=columns
        )
    # end if
# end csv_file


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


# Henon attractor
def henon(size, length, xy, a, b, washout=0, normalize=False, return_db=False, dtype=torch.float64):
    """
    Henon attractor
    """
    if return_db:
        return echotorch.datasets.HenonAttractor(
            sample_len=length,
            n_samples=size,
            xy=xy,
            a=a,
            b=b,
            washout=washout,
            normalize=normalize
        )
    else:
        return echotorch.datasets.HenonAttractor.generate(
            n_samples=size,
            sample_len=length,
            xy=xy,
            a=a,
            b=b,
            washout=washout,
            normalize=normalize,
            dtype=dtype
        )
    # end if
# end henon


# Mackey Glass timeseries
def mackey_glass(size, length, tau=17, return_db=False, dtype=torch.float64):
    """
    Mackey Glass timeseries
    """
    if return_db:
        return echotorch.datasets.MackeyGlassDataset(
            sample_len=length,
            n_samples=size,
            tau=tau
        )
    else:
        samples = list()
        for sample_i in range(size):
            return echotorch.datasets.MackeyGlassDataset.generate(

            )
        # end for
    # end if
# end mackey_glass


# NARMA
def narma(size, length, order=10, return_db=False, dtype=torch.float64):
    """
    NARMA-10
    """
    if return_db:
        return echotorch.datasets.NARMADataset(
            sample_len=length,
            n_samples=size,
            system_order=order
        )
    else:
        return echotorch.datasets.NARMADataset.generate(
            sample_len=length,
            n_samples=length,
            system_order=order,
            dtype=dtype
        )
    # end if
# end narma


# NARMA-10
def narma10(size, length, return_db=False, dtype=torch.float64):
    """
    NARMA-10
    """
    if return_db:
        return echotorch.datasets.NARMADataset(
            sample_len=length,
            n_samples=size,
            system_order=10
        )
    else:
        return echotorch.datasets.NARMADataset.generate(
            sample_len=length,
            n_samples=length,
            system_order=10,
            dtype=dtype
        )
    # end if
# end narma10


# NARMA-30
def narma30(size, length, return_db=False, dtype=torch.float64):
    """
    NARMA-30
    """
    if return_db:
        return echotorch.datasets.NARMADataset(
            sample_len=length,
            n_samples=size,
            system_order=30
        )
    else:
        return echotorch.datasets.NARMADataset.generate(
            sample_len=length,
            n_samples=length,
            system_order=30,
            dtype=dtype
        )
    # end if
# end narma30
