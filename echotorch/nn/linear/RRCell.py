# -*- coding: utf-8 -*-
#
# File : echotorch/nn/linear/RRCell.py
# Description : Ridge Regression node
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
# Copyright Nils Schaetti <nils.schaetti@unine.ch>

"""
Created on 26 January 2018
@author: Nils Schaetti
"""

# Imports
import torch.sparse
import torch
from ..Node import Node
from torch.autograd import Variable


# Ridge Regression node
class RRCell(Node):
    """
    Ridge Regression node
    """

    # Constructor
    def __init__(self, input_dim, output_dim, ridge_param=0.0, with_bias=True, learning_algo='inv',
                 softmax_output=False, normalize_output=False, averaged=True, debug=Node.NO_DEBUG, test_case=None,
                 dtype=torch.float32):
        """
        Constructor
        :param input_dim: Feature space dimension
        :param output_dim: Output space dimension
        :param ridge_param: Ridge parameter
        :param with_bias: Add a bias to the linear layer
        :param learning_algo: Inverse (inv) or pseudo-inverse (pinv)
        :param softmax_output: Add a softmax output (normalize outputs) ?
        :param normalize_output: Normalize outputs to sum to one ?
        :param averaged: Covariance matrix divided by the number of samples ?
        :param debug: Debug mode
        :param test_case: Test case to call for test.
        :param dtype: Data type
        """
        # Superclass
        super(RRCell, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            debug=debug,
            dtype=dtype
        )

        # Properties
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._ridge_param = ridge_param
        self._with_bias = with_bias
        self._learning_algo = learning_algo
        self._softmax_output = softmax_output
        self._normalize_output = normalize_output
        self._softmax = torch.nn.Softmax(dim=2)
        self._averaged = averaged
        self._n_samples = 0

        # Size
        if self._with_bias:
            self._x_size = input_dim + 1
        else:
            self._x_size = input_dim
        # end if

        # Set it as buffer
        self.register_buffer('xTx', Variable(torch.zeros(self._x_size, self._x_size, dtype=dtype), requires_grad=False))
        self.register_buffer('xTy', Variable(torch.zeros(self._x_size, output_dim, dtype=dtype), requires_grad=False))
        self.register_buffer('w_out', Variable(torch.zeros(output_dim, input_dim, dtype=dtype), requires_grad=False))
    # end __init__

    # region PROPERTIES

    # endregion PROPERTIES

    # region PUBLIC

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

        # Training or eval
        if self.training:
            for b in range(batch_size):
                if not self._averaged:
                    self.xTx.data.add_(x[b].t().mm(x[b]).data)
                    self.xTy.data.add_(x[b].t().mm(y[b]).data)
                else:
                    self.xTx.data.add_((x[b].t().mm(x[b]) / time_length).data)
                    self.xTy.data.add_((x[b].t().mm(y[b]) / time_length).data)
                    self._n_samples += 1.0
            # end for

            # Bias or not
            if self._with_bias:
                return x[:, :, 1:]
            else:
                return x
            # end if
        elif not self.training:
            # Outputs
            outputs = Variable(torch.zeros(batch_size, time_length, self._output_dim, dtype=self._dtype),
                               requires_grad=False)
            outputs = outputs.cuda() if self.w_out.is_cuda else outputs

            # For each batch
            for b in range(batch_size):
                outputs[b] = torch.mm(self.w_out, x[b].t()).t()
            # end for

            if self._softmax_output:
                return self._softmax(outputs)
            elif self._normalize_output:
                return torch.abs(outputs) / torch.sum(torch.abs(outputs), axis=2).reshape(outputs.size(0), outputs.size(1), 1)
            else:
                return outputs
        # end if

    # end forward

    # Finish training
    def finalize(self):
        """
        Finalize training with inverse or pseudo-inverse
        """
        if self._averaged:
            # Average
            self.xTx = self.xTx / self._n_samples
            self.xTy = self.xTy / self._n_samples
        # end if

        # We need to solve wout = (xTx)^(-1)xTy
        # Covariance matrix xTx
        ridge_xTx = self.xTx + self._ridge_param * torch.eye(self._input_dim + self._with_bias, dtype=self._dtype)

        # Inverse / pinverse
        if self._learning_algo == "inv":
            inv_xTx = ridge_xTx.inverse()
        elif self._learning_algo == "pinv":
            inv_xTx = ridge_xTx.pinverse()
        else:
            raise Exception("Unknown learning method {}".format(self._learning_algo))
        # end if

        # wout = (xTx)^(-1)xTy
        self.w_out = torch.mm(inv_xTx, self.xTy).t()

        # Not in training mode anymore
        self.train(False)
    # end finalize

    # endregion PUBLIC

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

    # region OVERRIDE

    # Extra-information
    def extra_repr(self):
        """
        Extra-information
        """
        s = super(RRCell, self).extra_repr()
        s += (', ridge_param={_ridge_param}, with_bias={_with_bias}, '
              'learning_algo={_learning_algo}, softmax_output={_softmax_output}, averaged={_averaged}')
        return s.format(**self.__dict__)
    # end extra_repr

    # endregion OVERRIDE

# end RRCell
