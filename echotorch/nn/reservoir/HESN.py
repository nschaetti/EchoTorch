# -*- coding: utf-8 -*-
#
# File : echotorch/nn/HESN.py
# Description : ESN with input pre-trained and used with transfer learning.
# Date : 22 March, 2018
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

import torch.sparse
from echotorch.nn.reservoir.LiESN import LiESN


# ESN with input pre-trained and used with transfer learning
class HESN(object):
    """
    ESN with input pre-trained and used with transfer learning
    """

    # Constructor
    def __init__(
            self, model, input_dim, hidden_dim, output_dim, spectral_radius=0.9,
            bias_scaling=0, input_scaling=1.0, w=None, w_in=None, w_bias=None, sparsity=None,
            input_set=[1.0, -1.0], w_sparsity=None, nonlin_func=torch.tanh, learning_algo='inv', ridge_param=0.0,
            leaky_rate=1.0, train_leaky_rate=False, feedbacks=False, wfdb_sparsity=None,
            normalize_feedbacks=False):
        # Embedding layer
        self.mode = model

        # Li-ESN
        self.esn = LiESN(
            input_dim, hidden_dim, output_dim, spectral_radius, bias_scaling, input_scaling,
            w, w_in, w_bias, sparsity, input_set, w_sparsity, nonlin_func, learning_algo, ridge_param,
            leaky_rate, train_leaky_rate, feedbacks, wfdb_sparsity, normalize_feedbacks
        )
    # end __init__

    # region PROPERTIES

    # Hidden layer
    @property
    def hidden(self):
        """
        Hidden layer
        :return:
        """
        return self.esn.hidden

    # end hidden

    # Hidden weight matrix
    @property
    def w(self):
        """
        Hidden weight matrix
        :return:
        """
        return self.esn.w

    # end w

    # Input matrix
    @property
    def w_in(self):
        """
        Input matrix
        :return:
        """
        return self.esn.w_in
    # end w_in

    # endregion PROPERTIES

    # region PUBLIC

    # endregion PUBLIC

    # region OVERRIDE

    # Forward
    def forward(self, u, y=None):
        """
        Forward
        :param x:
        :return:
        """
        # Selected features
        selected_features = self.model(u)

        # ESN
        return self.esn(selected_features, y)
    # end forward

    # endregion OVERRIDE

# end HESN
