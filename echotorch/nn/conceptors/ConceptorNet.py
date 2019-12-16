# -*- coding: utf-8 -*-
#
# File : echotorch/nn/ESN.py
# Description : An Echo State Network module.
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

"""
Created on 26 January 2018
@author: Nils Schaetti
"""

# Imports
import torch
from ..reservoir import ESN
from .SPESN import SPESN


# Conceptor Network
class ConceptorNet(SPESN):
    """
    Conceptor Network
    """
    # region BODY

    # Morphing type
    MORPHING_TYPE_TIMELESS = 0
    MORPHING_TYPE_TIME = 1

    # Constructor
    def __init__(self, conceptor, *args, **kwargs):
        """
        Constructor
        :param conceptor: Conceptor or ConceptorSet object
        :param args: Arguments
        :param kwargs: Additional arguments
        """
        super(ConceptorNet, self).__init__(
            *args,
            **kwargs
        )

        # Properties
        self._conceptor_active = False

        # Morphing vectors and counters
        self._morphing_type = None
        self._morphing_on = False
        self._morphing_vectors = None

        # Current conceptor
        self.conceptor = conceptor

        # Neural filter
        self._esn_cell.connect("neural-filter", self._neural_filter)
        self._esn_cell.connect("post-states-update", self._post_update_states)
    # end __init__

    # region PUBLIC

    # Set the current conceptor
    def set_conceptor(self, C):
        """
        Set the current conceptor
        :param C: The conceptor matrix
        """
        self.conceptor = C
    # end set_conceptor

    # Use conceptor ?
    def conceptor_active(self, value):
        """
        Use conceptors ?
        :param value: True/False
        """
        self._conceptor_active = value
    # end conceptor_active

    # endregion PUBLIC

    # region PRIVATE

    # Neural filter / training
    def _neural_filter(self, x, ut, forward_i, sample_i, t, washout):
        """
        Neural filter
        :param x: States to filter
        :param ut: Inputs
        :param forward_i: Forward call
        :param sample_i: Sample index (inside the batch)
        :param t: Time t
        :param washout: In washout period
        """
        if self._conceptor_active and self.conceptor is not None and not self.conceptor.training:
            # Morphing
            if self._morphing_on:
                # Current morphing vector
                if self._morphing_type == ConceptorNet.MORPHING_TYPE_TIMELESS:
                    morphing_vector = self._morphing_vectors[sample_i]
                else:
                    morphing_vector = self._morphing_vectors[sample_i, t]
                # end if

                # Conceptor filtering
                return self.conceptor(x, morphing_vector=morphing_vector)
            else:
                return self.conceptor(x)
            # end if
        else:
            return x
        # end if
    # end _neural_filter

    # Get states after batch update to train conceptors
    def _post_update_states(self, states, inputs, forward_i, sample_i):
        """
        Get states after batch update to train conceptors
        :param states: Reservoir states (without washout)
        :param inputs: Input signal
        :param forward_i: Forward call
        :param sample_i: Position in the batch
        """
        if self._conceptor_active and self.conceptor is not None and self.conceptor.training:
            self.conceptor(states)
        # end if
    # end _post_update_states

    # endregion PRIVATE

    # region OVERRIDE

    # Forward
    def forward(self, u, y=None, reset_state=True, morphing_vectors=None):
        """
        Forward
        :param u: Input signal
        :param y: Target outputs (or None if prediction)
        :param reset_state: Reset state before running layer (to zero)
        :param morphing_vectors: Morphing vectors (batch, time, number of conceptors)
        :return: Output (eval) or hidden states (training)
        """
        if morphing_vectors is not None:
            # Save morphing vectors
            self._morphing_vectors = morphing_vectors

            # Timeless morphing vectors ?
            if morphing_vectors.ndim == 2:
                self._morphing_type = ConceptorNet.MORPHING_TYPE_TIMELESS
            elif morphing_vectors.ndim == 3:
                self._morphing_type = ConceptorNet.MORPHING_TYPE_TIME
            else:
                raise Exception(
                    "Morphing vectors should have dimension 2 (timeless) or 3 (time) but has {}".format(
                        morphing_vectors.ndim
                    )
                )
            # end if

            # u and morphing vectors should have the same number of batch
            if u.size(0) != morphing_vectors.size(0):
                raise Exception(
                    "Inputs and morphing vectors have different batch sizes ({} != {})".format(
                        u.size(0),
                        morphing_vectors.size(0)
                    )
                )
            # end if

            # If not timeless, should have same time length
            if morphing_vectors.ndim == 3 and u.size(1) != morphing_vectors.size(1):
                raise Exception(
                    "Inputs and morphing vectors have different time length ({} != {})".format(
                        u.size(1),
                        morphing_vectors.size(1)
                    )
                )
            # end if

            # Init counter
            self._morphing_on = True

            # Forward from ESN
            return_esn = ESN.forward(self, u, y, reset_state)

            # Remove morphing vectors
            self._morphing_on = False
            self._morphing_vectors = None
            self._morphing_type = None

            return return_esn
        else:
            # Forward from ESN
            return ESN.forward(self, u, y, reset_state)
        # end if
    # end forward

    # Extra-information
    def extra_repr(self):
        """
        Extra-information
        :return: String
        """
        s = super(ConceptorNet, self).extra_repr()
        return s.format(**self.__dict__)
    # end extra_repr

    # endregion OVERRIDE

    # endregion BODY
# end ConceptorNet
