# -*- coding: utf-8 -*-
#
# File : echotorch/matrices.py
# Description : EchoTorch matrix creation utility functions.
# Date : 30th of March, 2021
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
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>,
# University of Geneva <nils.schaetti@unige.ch>


# Imports
from torch.optim import Optimizer


# RidgeRegression
class RidgeRegression(Optimizer):
    """
    Ridge Regression (RR) optimizer
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, params, hidden_dim, output_dim, ridge_param=0.0):
        """
        Constructor
        """
        # Check ridge param
        if ridge_param < 0:
            raise ValueError("Invalid ridge parameter: {}".format(ridge_param))
        # end if

        # Default parameter
        defaults = dict(ridge_param=ridge_param)

        # Super call
        super(RidgeRegression, self).__init__(params, defaults)
    # end __init__

    # endregion CONSTRUCTORS

# end RidgeRegression
