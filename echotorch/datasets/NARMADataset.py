# -*- coding: utf-8 -*-
#
# File : echotorch/datasets/NARMADataset.py
# Description : NARMA timeseries
# Date : 10th of September, 2020
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
import torch
from torch.utils.data.dataset import Dataset


# 10th order NARMA task
class NARMADataset(Dataset):
    """
    xth order NARMA task
    WARNING: this is an unstable dataset. There is a small chance the system becomes
    unstable, leading to an unusable dataset. It is better to use NARMA30 which
    where this problem happens less often.
    """

    # Constructor
    def __init__(self, sample_len, n_samples, system_order=10):
        """
        Constructor
        :param sample_len: Length of the time-series in time steps.
        :param n_samples: Number of samples to generate.
        :param system_order: th order NARMA
        """
        # Properties
        self.sample_len = sample_len
        self.n_samples = n_samples
        self.system_order = system_order

        # System order
        self.parameters = torch.zeros(4)
        if system_order == 10:
            self.parameters[0] = 0.3
            self.parameters[1] = 0.05
            self.parameters[2] = 9
            self.parameters[3] = 0.1
        else:
            self.parameters[0] = 0.2
            self.parameters[1] = 0.04
            self.parameters[2] = 29
            self.parameters[3] = 0.001
        # end if

        # Generate data set
        self.inputs, self.outputs = self._generate()
    # end __init__

    #############################################
    # OVERRIDE
    #############################################

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return self.n_samples
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        return self.inputs[idx], self.outputs[idx]
    # end __getitem__

    ##############################################
    # PRIVATE
    ##############################################

    # Generate
    def _generate(self):
        """
        Generate dataset
        :return:
        """
        inputs = list()
        outputs = list()
        for i in range(self.n_samples):
            ins = torch.rand(self.sample_len, 1) * 0.5
            outs = torch.zeros(self.sample_len, 1)
            for k in range(self.system_order - 1, self.sample_len - 1):
                outs[k + 1] = self.parameters[0] * outs[k] + self.parameters[1] * outs[k] * torch.sum(
                    outs[k - (self.system_order - 1):k + 1]) + 1.5 * ins[k - int(self.parameters[2])] * ins[k] + \
                                 self.parameters[3]
            # end for
            inputs.append(ins)
            outputs.append(outs)
        # end for

        return inputs, outputs
    # end _generate

# end NARMADataset
