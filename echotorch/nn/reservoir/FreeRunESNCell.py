# -*- coding: utf-8 -*-
#
# File : echotorch/nn/FreeRunESNCell.py
# Description : ESN Cell with Feedbacks
# Date : 28th of November, 2019
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

"""
Created on 28 November 2019
@author: Nils Schaetti
"""

# Imports
import torch
import torch.sparse
from torch.autograd import Variable

import echotorch.utils
from echotorch.utils.visualisation import Observable
from .LiESNCell import LiESNCell


# Echo State Network layer
# with Feedback
# TODO: Test
class FreeRunESNCell(LiESNCell, Observable):
    """
    Echo State Network layer
    Basis cell for ESN
    """

    # Constructor
    def __init__(self, feedback_noise, *args, **kwargs):
        """
        Constructor
        :param args: Arguments
        :param kwargs: Positional arguments
        """
        # Superclass
        super(FreeRunESNCell, self).__init__(*args, **kwargs)

        # Feedbacks matrix
        self._feedback_noise = feedback_noise
        self._w_fdb = None
    # end __init__

    # region PROPERTIES

    # endregion PROPERTIES

    # region PUBLIC

    # Set feedbacks
    def set_feedbacks(self, w_fdb):
        """
        Set feedbacks
        :param w_fdb: Feedback matrix (reservoir x
        """
        self._w_fdb = w_fdb
    # end set_feedbacks

    # endregion PUBLIC

    # region OVERRIDE

    # Hook which gets executed before the update state equation for every timesteps.
    def _pre_step_update_hook(self, inputs, forward_i, sample_i, t):
        """
        Hook which gets executed before the update equation for every timesteps
        :param inputs: Input signal.
        :param forward_i: Index of forward call
        :param sample_i: Position of the sample in the batch.
        :param t: Timestep.
        """
        if self._w_fdb is not None:
            return torch.mv(self._w_fdb, self.hidden) + self._feedback_noise(self._output_dim)
        else:
            return inputs
        # end if
    # end _pre_step_update_hook

    # endregion OVERRIDE

# end FreeRunESNCell
