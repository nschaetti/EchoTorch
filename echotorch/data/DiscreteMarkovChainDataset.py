# -*- coding: utf-8 -*-
#
# File : datasets/DiscreteMarkovChainDataset.py
# Description : Create discrete samples from a Markov Chain
# Date : 20th of December, 2019
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
from typing import List, Tuple
import torch
import torch.distributions.multinomial
import numpy as np

# Local imports
from .EchoDataset import EchoDataset


# Discrete Markov chain dataset
class DiscreteMarkovChainDataset(EchoDataset):
    """
    Discrete Markov chain dataset
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, n_samples, sample_length, probability_matrix, *args, **kwargs):
        """
        Constructor
        :param n_samples: Number of samples to generate
        :param sample_length: Number of steps to generate for each samples
        :param probability_matrix: Markov chain's probability matrix
        """
        # Super
        super(DiscreteMarkovChainDataset, self).__init__(*args, **kwargs)

        # Properties
        self._probability_matrix = probability_matrix
        self._n_samples = n_samples
        self._sample_length = sample_length
        self._n_states = probability_matrix.size(0)
    # end __init__

    # endregion CONSTRUCTORS

    # region PRIVATE

    # Generate a markov chain from a probability matrix
    def _generate_markov_chain(self, length, start_state=0):
        """
        Generate a sample from a probability matrix
        :param length: Length of the sample to generate
        :param start_state: Starting state
        """
        return self.datafunc(
            length=length,
            n_states=self._n_states,
            probability_matrix=self._probability_matrix,
            start_state=start_state
        )
    # end _generate_markov_chain

    # endregion PRIVATE

    # region OVERRIDE

    # Get the whole dataset
    @property
    def data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Get the whole dataset (according to init parameters)
        @return: The Torch Tensor
        """
        # List of x and y
        samples_in = list()
        samples_out = list()

        # For each samples
        for idx in range(self._n_samples):
            states_vector, next_states_vector = self[idx]
            samples_in.append(states_vector)
            samples_out.append(next_states_vector)
        # end for

        return samples_in, samples_out
    # end data

    # Extra representation
    def extra_repr(self) -> str:
        """
        Extra representation
        """
        return "n_samples={}, sample_length={}, probability_matrix={}, n_states={}".format(
            self._n_samples,
            self._sample_length,
            self._probability_matrix,
            self._n_states
        )
    # end extra_repr

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return self._n_samples
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx: Sample index
        :return: Sample as torch tensor
        """
        # Generate a Markov chain with
        # specified length.
        return self._generate_markov_chain(
            length=self._sample_length,
            start_state=np.random.randint(low=0, high=self._n_states-1)
        )
    # end __getitem__

    # Generate
    def datafunc(self, length, n_states, probability_matrix, start_state=0, dtype=torch.float64):
        """
        Generate
        :param length: Length
        :param n_states: How many states
        :param start_state: Starting state
        :param probability_matrix: Markov probability matrix
        """
        # One-hot vector of states
        states_vector = torch.zeros(length, n_states, dtype=dtype)
        states_vector[0, start_state] = 1.0

        # Next state to predict
        next_states_vector = torch.zeros(length, n_states, dtype=dtype)

        # Current state
        current_state = start_state

        # For each time step
        for t in range(1, length + 1):
            # Probability to next states
            prob_next_states = probability_matrix[current_state]

            # Create a multinomial distribution from probs.
            mnd = torch.distributions.multinomial.Multinomial(
                total_count=n_states,
                probs=prob_next_states
            )

            # Generate next states from probs.
            next_state = torch.argmax(mnd.sample()).item()

            # Save state
            if t < length:
                states_vector[t, next_state] = 1.0
            # end if

            # Save prediction
            next_states_vector[t - 1, next_state] = 1.0

            # Set as current
            current_state = next_state
        # end for

        return states_vector, next_states_vector
    # end datafunc

    # endregion OVERRIDE

# end DiscreteMarkovChainDataset
