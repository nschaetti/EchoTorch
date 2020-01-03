# -*- coding: utf-8 -*-
#
# File : echotorch/nn/linear/IncRRCell.py
# Description : Incremental Ridge Regression node
# Date : 5th of November, 2019
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
Created on 5 November 2019
@author: Nils Schaetti
"""

# Imports
import torch.sparse
import torch
from ..Node import Node
from torch.autograd import Variable


# Incremental Ridge Regression node
class IncRRCell(Node):
    """
    Incremental Ridge Regression node
    """

    # Constructor
    def __init__(self, input_dim, output_dim, conceptors, ridge_param=0.0, with_bias=False, learning_algo='pinv',
                 softmax_output=False, averaged=True, debug=Node.NO_DEBUG, test_case=None, dtype=torch.float32):
        """
        Constructor
        :param input_dim: Feature space dimension
        :param output_dim: Output space dimension
        :param conceptors: ConceptorSet object of conceptors used to describe space.
        :param ridge_param: Ridge parameter
        :param with_bias: Add a bias to the linear layer
        :param learning_algo: Inverse (inv) or pseudo-inverse (pinv)
        :param softmax_output: Add a softmax output (normalize outputs) ?
        :param averaged: Covariance matrix divided by the number of samples ?
        :param debug: Debug mode
        :param test_case: Test case to call for test.
        :param dtype: Data type
        """
        # Superclass
        super(IncRRCell, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            debug=debug,
            test_case=test_case,
            dtype=dtype
        )

        # Properties
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._conceptors = conceptors
        self._ridge_param = ridge_param
        self._with_bias = with_bias
        self._learning_algo = learning_algo
        self._softmax_output = softmax_output
        self._softmax = torch.nn.Softmax(dim=2)
        self._averaged = averaged
        self._n_samples = 0

        # Size
        if self._with_bias:
            self._x_size = input_dim + 1
        else:
            self._x_size = input_dim
        # end if

        # Wout matrix
        self.register_buffer('w_out', Variable(torch.zeros(1, input_dim, dtype=dtype), requires_grad=False))
        self.register_buffer('w_out_inc', Variable(torch.zeros(1, self.input_dim, dtype=self.dtype), requires_grad=False))
    # end __init__

    # region PRIVATE

    # Compute Wout increment
    def _compute_increment(self, X, Y, ridge_param, F=None):
        """
        Compute Wout increment
        """
        # Time length
        time_length = X.size()[0]

        # Targets to be learn : what is not predicted
        # by current Wout matrix
        Yt = Y - torch.mm(self.w_out, X.t()).t()

        # Debug
        self._call_debug_point("X{}".format(self._n_samples), X, "IncRRCell", "_compute_increment")
        self._call_debug_point("y{}".format(self._n_samples), Y, "IncRRCell", "_compute_increment")
        self._call_debug_point("Yt{}".format(self._n_samples), Yt, "IncRRCell", "_compute_increment")

        # Filter if the conceptor is not null
        if not self._conceptors.is_null():
            # The linear subspace of the reservoir state space that are not yet
            # occupied by any pattern.
            if F is None:
                F = self._conceptors.F()
            # end if

            # Debug
            self._call_debug_point("F{}".format(self._n_samples), F, "IncRRCell", "_compute_increment")

            # Filter training states to get what is new in the reservoir space
            S = torch.mm(F, X.t()).t()
        else:
            # No filter
            S = X
        # end if

        # Debug
        self._call_debug_point("S{}".format(self._n_samples), S, "IncRRCell", "_compute_increment")

        # sTs
        if self._averaged:
            sTs = torch.mm(S.t(), S) / time_length
        else:
            sTs = torch.mm(S.t(), S)
        # end if

        # Debug
        self._call_debug_point("sTs{}".format(self._n_samples), sTs, "IncRRCell", "_compute_increment")

        # sTy
        if self._averaged:
            sTy = torch.mm(S.t(), Yt) / time_length
        else:
            sTy = torch.mm(S.t(), Yt)
        # end if

        # Debug
        self._call_debug_point("sTy{}".format(self._n_samples), sTy, "IncRRCell", "_compute_increment")

        # Ridge sTs
        ridge_sTs = sTs + ridge_param * torch.eye(self._input_dim)

        # Debug
        self._call_debug_point("ridge_sTs{}".format(self._n_samples), ridge_sTs, "IncRRCell", "_compute_increment")

        # Inverse / pinverse
        if self._learning_algo == "inv":
            inv_sTs = self._inverse("ridge_sTs", ridge_sTs, "IncRRCell", "_compute_increment")
        elif self._learning_algo == "pinv":
            inv_sTs = self._pinverse("ridge_sTs", ridge_sTs, "IncRRCell", "_compute_increment")
        else:
            raise Exception("Unknown learning method {}".format(self._learning_algo))
        # end if

        # Debug
        self._call_debug_point("inv_sTs{}".format(self._n_samples), inv_sTs, "IncRRCell", "_compute_increment")

        # Compute increment for Wout
        Wout_inc = (torch.mm(inv_sTs, sTy)).t()

        # Debug
        self._call_debug_point("Wout_inc{}".format(self._n_samples), Wout_inc, "IncRRCell", "_compute_increment")

        return Wout_inc
    # end _compute_increment

    # Update Wout matrix
    def _update_Wout_loading(self, x, y):
        """
        Update Wout matrix
        """
        # Compute increment for Wout
        self.w_out_inc = self._compute_increment(x, y, self._ridge_param)

        # Increment Wout
        self.w_out += self.w_out_inc
    # end _update_Wout_loading

    # endregion PRIVATE

    # region OVERRIDE

    # Reset learning
    def reset(self):
        """
        Reset learning
        :return:
        """
        self.xTx.data.fill_(0.0)
        self.xTy.data.fill_(0.0)
        self.w_out.data.fill_(0.0)
        self._n_samples = 0

        # Training mode again
        self.train(True)
    # end reset

    # Forward
    def forward(self, x, y=None):
        """
        Forward
        :param x: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        # Batch size
        batch_size = x.size()[0]

        # Time length
        time_length = x.size()[1]

        # Add bias
        if self._with_bias:
            x = self._add_constant(x)
        # end if

        # State or output
        if self.training:
            # For each sample in the batch
            for b in range(batch_size):
                # Update Wout matrix
                self._update_Wout_loading(x[b], y[b])

                # Debug
                self._call_debug_point("w_out{}".format(self._n_samples), self.w_out, "IncRRCell", "forward")

                # One more sample
                self._n_samples += 1
            # end for

            # State
            return x
        else:
            # Outputs
            outputs = Variable(torch.zeros(batch_size, time_length, self._output_dim, dtype=self._dtype), requires_grad=False)
            outputs = outputs.cuda() if self.w_out.is_cuda else outputs

            # For each sample in the batch
            for b in range(batch_size):
                # Predicted output
                outputs[b] = torch.mm(self.w_out, x[b].t()).t()
            # end for

            # Softmax output ?
            if self._softmax_output:
                return self.softmax(outputs)
            else:
                return outputs
            # end if
        # end if
    # end forward

    # endregion OVERRIDE

    # region PRIVATE

    # Add constant
    def _add_constant(self, x):
        """
        Add constant
        :param x:
        :return:
        """
        if x.is_cuda:
            bias = Variable(torch.ones((x.size()[0], x.size()[1], 1), dtype=self.dtype).cuda(), requires_grad=False)
        else:
            bias = Variable(torch.ones((x.size()[0], x.size()[1], 1), dtype=self.dtype), requires_grad=False)
        # end if
        return torch.cat((bias, x), dim=2)
    # end _add_constant

    # endregion PRIVATE

# end IncRRCell
