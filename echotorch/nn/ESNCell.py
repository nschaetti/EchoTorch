# -*- coding: utf-8 -*-
#
# File : echotorch/nn/ESNCell.py
# Description : An Echo State Network layer.
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

import torch
import torch.sparse
from torch.autograd import Variable
import torch.nn as nn
import echotorch.utils
import numpy as np


# Echo State Network layer
class ESNCell(nn.Module):
    """
    Echo State Network layer
    """

    # Constructor
    def __init__(self, input_dim, output_dim, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0, w=None, w_in=None,
                 w_bias=None, w_fdb=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None,
                 nonlin_func=torch.tanh, feedbacks=False, feedbacks_dim=None, wfdb_sparsity=None,
                 normalize_feedbacks=False):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param output_dim: Reservoir size
        :param spectral_radius: Reservoir's spectral radius
        :param bias_scaling: Scaling of the bias, a constant input to each neuron (default: 0, no bias)
        :param input_scaling: Scaling of the input weight matrix, default 1.
        :param w: Internation weights matrix
        :param w_in: Input-reservoir weights matrix
        :param w_bias: Bias weights matrix
        :param sparsity:
        :param input_set:
        :param w_sparsity:
        :param nonlin_func: Reservoir's activation function (tanh, sig, relu)
        """
        super(ESNCell, self).__init__()

        # Params
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.bias_scaling = bias_scaling
        self.input_scaling = input_scaling
        self.sparsity = sparsity
        self.input_set = input_set
        self.w_sparsity = w_sparsity
        self.nonlin_func = nonlin_func
        self.feedbacks = feedbacks
        self.feedbacks_dim = feedbacks_dim
        self.wfdb_sparsity = wfdb_sparsity
        self.normalize_feedbacks = normalize_feedbacks

        # Init hidden state
        self.register_buffer('hidden', self.init_hidden())

        # Initialize input weights
        self.register_buffer('w_in', self._generate_win(w_in))

        # Initialize reservoir weights randomly
        self.register_buffer('w', self._generate_w(w))

        # Initialize bias
        self.register_buffer('w_bias', self._generate_wbias(w_bias))

        # Initialize feedbacks weights randomly
        if feedbacks:
            self.register_buffer('w_fdb', self._generate_wfdb(w_fdb))
        # end if
    # end __init__

    ###############################################
    # PUBLIC
    ###############################################

    # Forward
    def forward(self, u, y=None, w_out=None):
        """
        Forward
        :param u: Input signal
        :param y: Target output signal for teacher forcing
        :param w_out: Output weights for teacher forcing
        :return: Resulting hidden states
        """
        # Time length
        time_length = int(u.size()[1])

        # Number of batches
        n_batches = int(u.size()[0])

        # Outputs
        outputs = Variable(torch.zeros(n_batches, time_length, self.output_dim))
        outputs = outputs.cuda() if self.hidden.is_cuda else outputs

        # For each batch
        for b in range(n_batches):
            # Reset hidden layer
            self.reset_hidden()

            # For each steps
            for t in range(time_length):
                # Current input
                ut = u[b, t]

                # Compute input layer
                u_win = self.w_in.mv(ut)

                # Apply W to x
                x_w = self.w.mv(self.hidden)

                # Feedback or not
                if self.feedbacks and self.training and y is not None:
                    # Current target
                    yt = y[b, t]

                    # Compute feedback layer
                    y_wfdb = self.w_fdb.mv(yt)

                    # Add everything
                    x = u_win + x_w + y_wfdb + self.w_bias
                elif self.feedbacks and not self.training and w_out is not None:
                    # Compute past output
                    yt = w_out.mv(self.hidden)

                    # Normalize
                    if self.normalize_feedbacks:
                        yt -= torch.min(yt)
                        yt /= torch.max(yt) - torch.min(yt)
                        yt /= torch.sum(yt)
                    # end if

                    # Compute feedback layer
                    y_wfdb = self.w_fdb.mv(yt)

                    # Add everything
                    x = u_win + x_w + y_wfdb + self.w_bias
                else:
                    # Add everything
                    x = u_win + x_w + self.w_bias
                # end if

                # Apply activation function
                x = self.nonlin_func(x)

                # Add to outputs
                self.hidden.data = x.view(self.output_dim).data

                # New last state
                outputs[b, t] = self.hidden
            # end for
        # end for

        return outputs
    # end forward

    # Init hidden layer
    def init_hidden(self):
        """
        Init hidden layer
        :return: Initiated hidden layer
        """
        return Variable(torch.zeros(self.output_dim), requires_grad=False)
        # return torch.zeros(self.output_dim)
    # end init_hidden

    # Reset hidden layer
    def reset_hidden(self):
        """
        Reset hidden layer
        :return:
        """
        self.hidden.fill_(0.0)
    # end reset_hidden

    # Get W's spectral radius
    def get_spectral_radius(self):
        """
        Get W's spectral radius
        :return: W's spectral radius
        """
        return echotorch.utils.spectral_radius(self.w)
    # end spectral_radius

    ###############################################
    # PRIVATE
    ###############################################

    # Generate W matrix
    def _generate_w(self, w):
        """
        Generate W matrix
        :return:
        """
        # Initialize reservoir weight matrix
        if w is None:
            w = self.generate_w(self.output_dim, self.w_sparsity)
        else:
            if callable(w):
                w = w(self.output_dim)
            # end if
        # end if

        # Scale it to spectral radius
        w *= self.spectral_radius / echotorch.utils.spectral_radius(w)

        return Variable(w, requires_grad=False)
    # end generate_W

    # Generate Win matrix
    def _generate_win(self, w_in):
        """
        Generate Win matrix
        :return:
        """
        # Initialize input weight matrix
        if w_in is None:
            if self.sparsity is None:
                w_in = self.input_scaling * (
                            np.random.randint(0, 2, (self.output_dim, self.input_dim)) * 2.0 - 1.0)
                w_in = torch.from_numpy(w_in.astype(np.float32))
            else:
                w_in = self.input_scaling * np.random.choice(np.append([0], self.input_set),
                                                             (self.output_dim, self.input_dim),
                                                             p=np.append([1.0 - self.sparsity],
                                                                         [self.sparsity / len(self.input_set)] * len(
                                                                             self.input_set)))
                w_in = torch.from_numpy(w_in.astype(np.float32))
            # end if
        else:
            if callable(w_in):
                w_in = w_in(self.output_dim, self.input_dim)
            # end if
        # end if

        return Variable(w_in, requires_grad=False)
    # end _generate_win

    # Generate Wbias matrix
    def _generate_wbias(self, w_bias):
        """
        Generate Wbias matrix
        :return:
        """
        # Initialize bias matrix
        if w_bias is None:
            w_bias = self.bias_scaling * (torch.rand(1, self.output_dim) * 2.0 - 1.0)
        else:
            if callable((w_bias)):
                w_bias = w_bias(self.output_dim)
            # end if
        # end if

        return Variable(w_bias, requires_grad=False)
    # end _generate_wbias

    # Generate Wfdb matrix
    def _generate_wfdb(self, w_fdb):
        """
        Generate Wfdb matrix
        :return:
        """
        # Initialize feedbacks weight matrix
        if w_fdb is None:
            if self.wfdb_sparsity is None:
                w_fdb = self.input_scaling * (
                        np.random.randint(0, 2, (self.output_dim, self.feedbacks_dim)) * 2.0 - 1.0)
                w_fdb = torch.from_numpy(w_fdb.astype(np.float32))
            else:
                w_fdb = self.input_scaling * np.random.choice(np.append([0], self.input_set),
                                                             (self.output_dim, self.feedbacks_dim),
                                                             p=np.append([1.0 - self.wfdb_sparsity],
                                                                         [self.wfdb_sparsity / len(
                                                                             self.input_set)] * len(
                                                                             self.input_set)))
                w_fdb = torch.from_numpy(w_fdb.astype(np.float32))
            # end if
        else:
            if callable(w_fdb):
                w_fdb = w_fdb(self.output_dim, self.feedbacks_dim)
            # end if
        # end if

        return Variable(w_fdb, requires_grad=False)
    # end _generate_wfdb

    ############################################
    # STATIC
    ############################################

    # Generate W matrix
    @staticmethod
    def generate_w(output_dim, w_sparsity=None):
        """
        Generate W matrix
        :param output_dim:
        :param w_sparsity:
        :return:
        """
        # Sparsity
        if w_sparsity is None:
            w = torch.rand(output_dim, output_dim) * 2.0 - 1.0
        else:
            w = np.random.choice([0.0, 1.0], (output_dim, output_dim),
                                 p=[1.0 - w_sparsity, w_sparsity])
            w[w == 1] = np.random.rand(len(w[w == 1])) * 2.0 - 1.0
            w = torch.from_numpy(w.astype(np.float32))

            # Return
            return w
        # end if
        return w
    # end generate_w

    # To sparse matrix
    @staticmethod
    def to_sparse(m):
        """
        To sparse matrix
        :param m:
        :return:
        """
        # Rows, columns and values
        rows = torch.LongTensor()
        columns = torch.LongTensor()
        values = torch.FloatTensor()

        # For each row
        for i in range(m.shape[0]):
            # For each column
            for j in range(m.shape[1]):
                if m[i, j] != 0.0:
                    rows = torch.cat((rows, torch.LongTensor([i])), dim=0)
                    columns = torch.cat((columns, torch.LongTensor([j])), dim=0)
                    values = torch.cat((values, torch.FloatTensor([m[i, j]])), dim=0)
                # end if
            # end for
        # end for

        # Indices
        indices = torch.cat((rows.unsqueeze(0), columns.unsqueeze(0)), dim=0)

        # To sparse
        return torch.sparse.FloatTensor(indices, values)
    # end to_sparse

# end ESNCell
