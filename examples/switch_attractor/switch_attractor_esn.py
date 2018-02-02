# -*- coding: utf-8 -*-
#
# File : examples/SwitchAttractor/switch_attractor_esn
# Description : Attractor switching task with ESN.
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
import EchoTorch.nn as etnn
from torch.autograd import Variable


# Input size
input_dim = 2

# Reservoir size
n_hidden = 10

# ESN cell
esn = etnn.ESNCell(input_dim, n_hidden)

# Init hidden
hidden = esn.init_hidden()

# Inputs
inputs = Variable(torch.rand(input_dim))

print(u"init: {}".format(hidden))
print(u"inputs : {}".format(inputs))
print(u"win : {}".format(esn.win))
print(u"w : {}".format(esn.w))
print(u"wbias : {}".format(esn.wbias))

# Compute next state
next_hidden = esn(inputs, hidden)
print(next_hidden)
print(u"##################################")
# Another state
next_hidden = esn(inputs, next_hidden)
print(next_hidden)