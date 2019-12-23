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
import math
import torch
import torch.distributions.multinomial
from torch.utils.data.dataset import Dataset
import numpy as np


# Discrete Markov chain dataset
class DiscreteMarkovChainDataset(Dataset):
    """
    Discrete Markov chain dataset
    """

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

    # region PRIVATE

    # Generate a markov chain from a probability matrix
    def _generate_markov_chain(self, length, start_state=0):
        """
        Generate a sample from a probability matrix
        :param length: Length of the sample to generate
        :param start_state: Starting state
        """
        # One-hot vector of states
        states_vector = torch.zeros(length, self._n_states)
        states_vector[0, start_state] = 1.0

        # Next state to predict
        next_states_vector = torch.zeros(length, self._n_states)

        # Current state
        current_state = start_state

        # For each time step
        for t in range(1, length + 1):
            # Probability to next states
            prob_next_states = self._probability_matrix[current_state]

            # Create a multinomial distribution from probs.
            mnd = torch.distributions.multinomial.Multinomial(
                total_count=self._n_states,
                probs=prob_next_states
            )

            # Generate next states from probs.
            next_state = torch.argmax(mnd.sample()).item()

            # Save state
            if t < length:
                states_vector[t, next_state] = 1.0
            # end if

            # Save prediction
            next_states_vector[t-1, next_state] = 1.0

            # Set as current
            current_state = next_state
        # end for

        return states_vector, next_states_vector
    # end _generate_markov_chain

    # endregion ENDPRIVATE

    # region OVERRIDE

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

    # endregion OVERRIDE

# end DiscreteMarkovChainDataset
