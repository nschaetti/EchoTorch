# -*- coding: utf-8 -*-
#
# File : examples/conceptors/boolean_operations.py
# Description : Conceptor boolean operation
# Date : 16th of December, 2019
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
import numpy as np
import torch
import torch.distributions.multinomial
from torch.utils.data.dataset import Dataset
import numpy as np


# Markov chain dataset from patterns
class MarkovChainDataset(Dataset):
    """
    Markov chain dataset from patterns
    """

    # Constructor
    def __init__(self, datasets, states_length, morphing_length, n_samples, sample_length, probability_matrix,
                 random_start=0, *args, **kwargs):
        """
        Constructor
        :param datasets: Datasets
        :param states_length: State length
        :param morphing_length: Morphing length between patterns
        """
        # Super
        super(MarkovChainDataset, self).__init__(*args, **kwargs)

        # Properties
        self._probability_matrix = probability_matrix
        self.states = datasets
        self.n_states = len(datasets)
        self._states_length = states_length
        self._morphing_length = morphing_length
        self._sample_length = sample_length
        self._total_length = sample_length * (states_length + 2 * morphing_length)
        self._random_start = random_start
        self.n_samples = n_samples
    # region PRIVATE

    # Generate a markov chain from a probability matrix
    def _generate_markov_chain(self, length, start_state=0):
        """
        Generate a sample from a probability matrix
        :param probability_matrix: Probability matrix of the Markov chain
        :param length: Length of the sample to generate
        :param start_state: Starting state
        """
        # Vector of states
        states_vector = torch.zeros(length)
        states_vector[0] = start_state

        # Current state
        current_state = start_state

        # For each time step
        for t in range(1, length):
            # Probability to next states
            prob_next_states = self._probability_matrix[current_state]

            # Create a multinomial distribution from probs.
            mnd = torch.distributions.multinomial.Multinomial(
                total_count=self.n_states,
                probs=prob_next_states
            )

            # Generate next states from probs.
            next_state = torch.argmax(mnd.sample()).item()

            # Save next state
            states_vector[t] = next_state

            # Set as current
            current_state = next_state
        # end for

        return states_vector
    # end _generate_markov_chain

    # endregion ENDPRIVATE

    # region OVERRIDE

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return self.n_samples
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        # Generate a Markov chain with
        # specified length
        markov_chain = self._generate_markov_chain(
            length=self._sample_length,
            start_state=np.random.randint(low=0, high=self.n_states-1)
        )

        # Length of each state with morphing
        state_lenght_and_morph = self._states_length + 2 * self._morphing_length

        # Length of sample to get for each state
        state_sample_length = self._states_length + 4 * self._morphing_length

        # Empty tensor for the result
        inputs = torch.zeros(self._total_length, 1)

        # Outputs with each state at time t
        outputs = torch.zeros(self._total_length, self.n_states)

        # Save the list of samples
        list_of_state_samples = list()

        # For each state in the Markov chain
        for state_i in range(self._sample_length):
            # State
            current_state = int(markov_chain[state_i].item())

            # Get state dataset
            state_dataset = self.states[current_state]

            # Get a random sample
            state_sample = state_dataset[np.random.randint(len(state_dataset))]

            # State sample's length and minimum length
            minimum_length = (state_sample_length + self._random_start)

            # Check that the sample has
            # the minimum size
            if state_sample.size(0) >= minimum_length:
                # Random start
                if self._random_start > 0:
                    random_start = np.random.randint(0, self._random_start)
                else:
                    random_start = 0
                # end if

                # Get a random part
                state_part = state_sample[random_start:random_start+state_sample_length]

                # Add to the list
                list_of_state_samples.append(state_part)
            else:
                raise Exception("State sample length is not enough ({} vs {})!".format(state_sample_length, minimum_length))
            # end if
        # end for

        # Go through time to compose the sample
        for t in range(self._total_length):
            # In which state we are
            state_step = math.floor(t / state_lenght_and_morph)
            last_step = state_step - 1
            next_step = state_step + 1

            # Bounds
            state_start_time = state_step * state_lenght_and_morph
            state_end_time = (state_step + 1) * state_lenght_and_morph

            # Position in the state
            state_position = t - state_start_time

            # Are we in the morphing period
            if self._morphing_length > 0:
                if t - state_start_time < self._morphing_length and state_step > 0:
                    m = -(1.0 / (2.0 * self._morphing_length)) * state_position + 0.5
                    last_state_position = state_sample_length - self._morphing_length + state_position
                    inputs[t] = m * list_of_state_samples[last_step][last_state_position]
                    inputs[t] += (1.0 - m) * list_of_state_samples[state_step][state_position]
                elif state_end_time - t < self._morphing_length and state_step != self._sample_length - 1:
                    m = (1.0 / (2.0 * self._morphing_length)) * (state_end_time - t)
                    next_state_position = state_sample_length - (self._morphing_length + self._states_length)
                    inputs[t] = m * list_of_state_samples[next_step][next_state_position]
                    inputs[t] += (1.0 - m) * list_of_state_samples[state_step][state_position]
                else:
                    inputs[t] = list_of_state_samples[state_step][state_position]
                # end if
            else:
                inputs[t] = list_of_state_samples[state_step][state_position]
            # end if

            # Outputs
            outputs[t, int(markov_chain[state_step].item())] = 1.0
        # end for

        return inputs, outputs, markov_chain
    # end __getitem__

    # endregion OVERRIDE

# end DatasetComposer
