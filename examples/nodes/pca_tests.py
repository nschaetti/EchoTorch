# -*- coding: utf-8 -*-
#
# File : examples/timeserie_prediction/switch_attractor_esn
# Description : NARMA 30 prediction with ESN.
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


# Imports
import torch
import echotorch.nn as etnn
from torch.autograd import Variable
import mdp


# Settings
input_dim = 10
output_dim = 3
tlen = 500

# Generate
training_samples = torch.randn(1, tlen, input_dim)
test_samples = torch.randn(1, tlen, input_dim)

# Generate
training_samples_np = training_samples[0].numpy()
test_samples_np = test_samples[0].numpy()

# Show
print(u"Training samples : {}".format(training_samples_np))
print(u"Test samples : {}".format(test_samples_np))

# PCA node
mdp_pca_node = mdp.Flow([mdp.nodes.PCANode(input_dim=input_dim, output_dim=output_dim)])
mdp_pca_node.train(training_samples_np)
pca_reduced = mdp_pca_node(test_samples_np)

# Show
print(u"PCA reduced : {}".format(pca_reduced))

# EchoTorch PCA node
et_pca_node = etnn.PCACell(input_dim=input_dim, output_dim=output_dim)
et_pca_node(Variable(training_samples))
et_pca_node.finalize()
et_reduced = et_pca_node(Variable(test_samples))

# Show
print(u"Reduced with EchoTorch/PCA :")
print(et_reduced)
