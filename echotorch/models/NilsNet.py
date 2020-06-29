# -*- coding: utf-8 -*-
#
# File : echotorch/models/NilsNet.py
# Description : A NilsNet module.
# Date : 09th of April, 2018
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

# Imports
import torchvision
import torch.nn as nn
import echotorch.nn as ecnn


# A NilsNet
class NilsNet(nn.Module):
    """
    A NilsNet
    """

    # Constructor
    def __init__(self, reservoir_dim, sfa_dim, ica_dim, pretrained=False, feature_selector='resnet18'):
        """
        Constructor
        """
        # Upper class
        super(NilsNet, self).__init__()

        # ResNet
        if feature_selector == 'resnet18':
            self.feature_selector = torchvision.models.resnet18(pretrained=True)
        elif feature_selector == 'resnet34':
            self.feature_selector = torchvision.models.resnet34(pretrained=True)
        elif feature_selector == 'resnet50':
            self.feature_selector = torchvision.models.resnet50(pretrained=True)
        elif feature_selector == 'alexnet':
            self.feature_selector = torchvision.models.alexnet(pretrained=True)
        # end if

        # Skip last layer
        self.reservoir_input_dim = self.feature_selector.fc.in_features
        self.feature_selector.fc = ecnn.Identity()

        # Echo State Network
        # self.esn = ecnn.ESNCell(input_dim=self.reservoir_input_dim, output_dim=reservoir_dim)

        # Slow feature analysis layer
        # self.sfa = ecnn.SFACell(input_dim=reservoir_dim, output_dim=sfa_dim)

        # Independent Feature Analysis layer
        # self.ica = ecnn.ICACell(input_dim=sfa_dim, output_dim=ica_dim)
    # end __init__

    # Forward
    def forward(self, x):
        """
        Forward
        :return:
        """
        # ResNet
        return self.feature_selector(x)
    # end forward

# end NilsNet
