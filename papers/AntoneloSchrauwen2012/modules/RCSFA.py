# -*- coding: utf-8 -*-
#
# File : papers/AntoneloSchrauwen2012/modules/RCSFA.py
# Description : Implement the RC-SFA module.
# Date : 16th of September, 2020
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
# Copyright Nils Schaetti <nils.schaetti@unine.ch>, <nils.schaetti@unige.ch>

# Imports
import torch
import echotorch.nn as etnn
import torch.nn as nn


# RC-SFA module
class RCSFA(etnn.ESN):
    """
    The RC-SFA module
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, sfa_dim, output_dim, w_generator, win_generator, wbias_generator,
                 input_scaling=1.0, nonlin_func=torch.tanh, washout=0, debug=etnn.Node.NO_DEBUG, test_case=None,
                 dtype=torch.float64):
        """
        Constructor
        :param input_dim:
        :param hidden_dim:
        :param sfa_dim:
        :param output_dim:
        :param w_generator:
        :param win_generator:
        :param wbias_generator:
        :param input_scaling:
        :param nonlin_func:
        :param washout:
        :param debug:
        :param dtype:
        """
        # Call upper class
        super(RCSFA, self).__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            w_generator=w_generator,
            win_generator=win_generator,
            wbias_generator=wbias_generator,
            input_scaling=input_scaling,
            nonlin_func=nonlin_func,
            washout=washout,
            create_rnn=True,
            create_output=False,
            debug=debug,
            test_case=test_case,
            dtype=dtype
        )

        # Properties
        self._sfa_dim = sfa_dim

        # Create SFA layer
        self._sfa_layer = etnn.SFACell(
            input_dim=hidden_dim,
            output_dim=sfa_dim
        )

        # Create ICA output layer
        self._output = etnn.ICACell(
            input_dim=sfa_dim,
            output_dim=output_dim
        )
    # end __init__

    # region PROPERTIES

    # SFA_cell
    @property
    def cell_sfa(self):
        """
        SFA cell
        :return: SFA cell
        """
        return self._sfa_layer
    # end cell

    # endregion PROPERTIES

    # region PUBLIC

    # endregion PUBLIC

    # region OVERRIDE

    # Forward
    def forward(self, u, y=None, reset_state=True):
        """
        Forward
        :param u: Input signal.
        :param y: Target outputs (or None if prediction)
        :return: Output or hidden states
        """
        # Compute hidden states
        hidden_states = self._esn_cell(u, reset_state=reset_state)

        # Learning algo
        if not self.training:
            if not self._sfa_layer.training:
                return self._sfa_layer(hidden_states)
            else:
                return self._output(self._sfa_layer(hidden_states))
            # end if
        else:
            return self._output(self._sfa_layer(hidden_states))
        # end if
    # end forward

    # Finish training
    def finalize(self):
        """
        Finish training
        """
        # If SFA not trained, finalize this one
        if not self._sfa_layer.training:
            self._sfa_layer.finalize()
        elif not self._output.training:
            # Finalize
            self._output.finalize()

            # In eval mode
            self.train(False)
        # end if
    # end finalize

    # endregion OVERRIDE

# end RCSFA
