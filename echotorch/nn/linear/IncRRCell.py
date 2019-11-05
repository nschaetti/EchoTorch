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
    def __init__(self, input_dim, output_dim, conceptors, ridge_param=0.0, with_bias=True, learning_algo='inv',
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
    # end __init__

    #####################
    # PROPERTIES
    #####################

    #####################
    # PUBLIC
    #####################

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

        # Outputs
        outputs = Variable(torch.zeros(batch_size, time_length, self._output_dim, dtype=self._dtype), requires_grad=False)
        outputs = outputs.cuda() if self.w_out.is_cuda else outputs

        # For each sample in the batch
        for b in range(batch_size):
            # Targets to be learn : what is not predicted
            # by current Wout matrix
            Tout = y - torch.mm(self.w_out, x)

            # The linear subspace of the reservoir state space that are not yet
            # occupied by any pattern.
            F = self._conceptors.F()

            # Filter training states to get what is new in the reservoir space
            S = torch.mm(F, x)

            # SS pseudo-inverse
            inv_sTs = torch.pinverse(torch.mm(S, S.t()) / time_length + self._ridge_param * torch.eye(self.input_dim))

            # Compute increment for Wout
            Wout_inc = (
                torch.mm(torch.mm(inv_sTs, S), Tout.t()) / time_length
            ).t()

            # Increment Wout
            self.w_out += Wout_inc

            # Predicted output
            outputs[b] = torch.mm(x[b], self.w_out)
            # end for
        # end for

        # Softmax output ?
        if self._softmax_output:
            return self.softmax(outputs)
        else:
            return outputs
        # end if
    # end forward

    ###############################################
    # PRIVATE
    ###############################################

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

# end IncRRCell
