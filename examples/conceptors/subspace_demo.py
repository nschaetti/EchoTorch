# -*- coding: utf-8 -*-
#
# File : examples/timeserie_prediction/narma10_esn
# Description : NARMA-10 prediction with ESN.
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
import numpy as np
import torch
import echotorch.nn.conceptors as ecnc
import echotorch.utils.matrix_generation as mg
import argparse
import echotorch.utils
import echotorch.datasets as etds
from echotorch.datasets import DatasetComposer
from echotorch.nn.Node import Node
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

# Debug ?
debug = False
if debug:
    torch.set_printoptions(precision=16)
    debug_mode = Node.DEBUG_OUTPUT
    precision = 0.000001
else:
    debug_mode = Node.NO_DEBUG
# end if

# Random numb. init
torch.random.manual_seed(1)
np.random.seed(1)

# ESN params
reservoir_size = 100
spectral_radius = 1.5
input_scaling = 1.5
bias_scaling = 0.2
connectivity = 10.0 / reservoir_size
dtype=torch.float64

# Sequence lengths
washout_length = 500
learn_length = 1000
signal_plot_length = 20
conceptor_test_length = 200
singular_plot_length = 50
free_run_length = 100000

# Regularization
ridge_param_wstar = 0.0001
ridge_param_wout = 0.01

# Aperture
alpha = 10
alphas = [1.0, 10.0, 100.0, 1000.0, 10000.0]

# Argument parsing
parser = argparse.ArgumentParser(prog="subspace_demo", description=u"Fig. 1 BC subspace first demo")
parser.add_argument("--w", type=str, default="", required=False)
parser.add_argument("--w-name", type=str, default="", required=False)
parser.add_argument("--win", type=str, default="", required=False)
parser.add_argument("--win-name", type=str, default="", required=False)
parser.add_argument("--wbias", type=str, default="", required=False)
parser.add_argument("--wbias-name", type=str, default="", required=False)
parser.add_argument("--x0", type=str, default="", required=False)
parser.add_argument("--x0-name", type=str, default="", required=False)
args = parser.parse_args()

# Load W from matlab file and random init ?
if args.w != "":
    # Load internal weights
    w_generator = mg.matrix_factory.get_generator("matlab", file_name=args.w, entity_name=args.w_name)
else:
    # Generate internal weights
    w_generator = mg.matrix_factory.get_generator("normal", mean=0.0, std=1.0, connectivity=connectivity)
# end if

# Load Win from matlab file or init randomly
if args.win != "":
    # Load internal weights
    win_generator = mg.matrix_factory.get_generator("matlab", file_name=args.win, entity_name=args.win_name)
else:
    # Generate Win
    win_generator = mg.matrix_factory.get_generator("normal", mean=0.0, std=1.0, connectivity=1.0)
# end if

# Load Wbias from matlab from or init randomly
if args.wbias != "":
    wbias_generator = mg.matrix_factory.get_generator("matlab", file_name=args.wbias, entity_name=args.wbias_name, shape=reservoir_size)
else:
    wbias_generator = mg.matrix_factory.get_generator("normal", mean=0.0, std=1.0, connectivity=1.0)
# end if

# Four pattern (two sine, two periodic)
pattern1_training = etds.SinusoidalTimeseries(sample_len=washout_length + learn_length, n_samples=1, a=1,
    period=8.8342522,
    dtype=dtype
)
pattern2_training = etds.SinusoidalTimeseries(sample_len=washout_length + learn_length, n_samples=1, a=1,
    period=9.8342522,
    dtype=dtype
)
pattern3_training = etds.PeriodicSignalDataset(sample_len=washout_length + learn_length, n_samples=1,
    period=[0.9000000000000002, -0.11507714997817164, 0.17591170369788622, -0.9, -0.021065045054201592],
    dtype=dtype
)
pattern4_training = etds.PeriodicSignalDataset(sample_len=washout_length + learn_length, n_samples=1,
    period=[0.9, -0.021439412841318672, 0.0379515995051003, -0.9, 0.06663989939293802],
    dtype=dtype
)

# Composer
dataset_training = DatasetComposer([pattern1_training, pattern2_training, pattern3_training, pattern4_training])

# Data loader
patterns_loader = DataLoader(dataset_training, batch_size=1, shuffle=False, num_workers=1)

# Create a self-predicting ESN
# which will be loaded with the
# four patterns.
spesn = ecnc.SPESN(
    input_dim=1,
    hidden_dim=reservoir_size,
    output_dim=1,
    spectral_radius=spectral_radius,
    learning_algo='inv',
    w_generator=w_generator,
    win_generator=win_generator,
    wbias_generator=wbias_generator,
    input_scaling=input_scaling,
    bias_scaling=bias_scaling,
    ridge_param=ridge_param_wout,
    w_ridge_param=ridge_param_wstar,
    washout=washout_length,
    debug=debug_mode,
    dtype=dtype
)

# Create a conceptor network using
# the self-predicting ESN which
# will learn four conceptors.
conceptor_net = ecnc.ConceptorNet(
    input_dim=1,
    hidden_dim=reservoir_size,
    output_dim=1,
    esn_cell=spesn.cell
)

# If in debug mode
if debug_mode > Node.NO_DEBUG:
    # Load sample matrices
    for i in range(4):
        # Input patterns
        spesn.cell.debug_point(
            "u{}".format(i),
            torch.reshape(torch.from_numpy(np.load("data/debug/subspace_demo/u{}.npy".format(i))), shape=(-1, 1)),
            precision
        )

        # States
        spesn.cell.debug_point(
            "X{}".format(i),
            torch.from_numpy(np.load("data/debug/subspace_demo/X{}.npy".format(i))),
            precision
        )

        # Targets
        spesn.cell.debug_point(
            "Y{}".format(i),
            torch.from_numpy(np.load("data/debug/subspace_demo/Y{}.npy".format(i))),
            precision
        )

        # Xold
        spesn.cell.debug_point(
            "Xold{}".format(i),
            torch.from_numpy(np.load("data/debug/subspace_demo/Xold{}.npy".format(i))),
            precision
        )
    # end for

    # Load debug W, xTx, xTy
    spesn.cell.debug_point("Wstar", torch.from_numpy(np.load("data/debug/subspace_demo/Wstar.npy", allow_pickle=True)), precision)
    spesn.cell.debug_point("Win", torch.from_numpy(np.load("data/debug/subspace_demo/Win.npy")), precision)
    spesn.cell.debug_point("Wbias", torch.from_numpy(np.load("data/debug/subspace_demo/Wbias.npy")), precision)
    spesn.cell.debug_point("xTx", torch.from_numpy(np.load("data/debug/subspace_demo/xTx.npy")), precision)
    spesn.cell.debug_point("xTy", torch.from_numpy(np.load("data/debug/subspace_demo/xTy.npy")), precision)
    spesn.cell.debug_point("w_ridge_param", 0.0001, precision)
    spesn.cell.debug_point("ridge_xTx", torch.from_numpy(np.load("data/debug/subspace_demo/ridge_xTx.npy")), precision)
    spesn.cell.debug_point("inv_xTx", torch.from_numpy(np.load("data/debug/subspace_demo/inv_xTx.npy")), precision)
    spesn.cell.debug_point("w", torch.from_numpy(np.load("data/debug/subspace_demo/W.npy")), precision)
# end if

# Xold and Y collectors
Xold_collector = torch.empty(4 * learn_length, reservoir_size, dtype=dtype)
Y_collector = torch.empty(4 * learn_length, reservoir_size, dtype=dtype)

# Create four conceptors, one for each pattern
conceptors = [
    ecnc.Conceptor(input_dim=reservoir_size, aperture=alpha),
    ecnc.Conceptor(input_dim=reservoir_size, aperture=alpha),
    ecnc.Conceptor(input_dim=reservoir_size, aperture=alpha),
    ecnc.Conceptor(input_dim=reservoir_size, aperture=alpha)
]

# Go through dataset
for i, data in enumerate(patterns_loader):
    # Inputs and labels
    inputs, outputs, labels = data

    # Set current conceptor
    conceptor_net.set_conceptor(conceptors[i])

    # Feed SP-ESN
    X = conceptor_net(inputs, inputs)

    # Get targets
    Y = conceptor_net.cell.targets(X[0])

    # Get features
    Xold = conceptor_net.cell.features(X[0])

    # Save
    Xold_collector[i*learn_length:i*learn_length+learn_length] = Xold
    Y_collector[i*learn_length:i*learn_length+learn_length] = Y
# end for

# Finalize training
conceptor_net.finalize()

# Predicted by W
predY = torch.mm(conceptor_net.cell.w, Xold_collector.t()).t()

# Compute NRMSE
training_NRMSE = echotorch.utils.nrmse(predY, Y_collector)
print("Training NRMSE : {}".format(training_NRMSE))

# Run trained ESN with empty inputs and plot it
generated = conceptor_net(torch.zeros(1, conceptor_test_length, 1, dtype=dtype))
plt.plot(generated[0])
plt.show()

# Save each generated pattern for display
generated_samples = torch.zeros(4, conceptor_test_length)

# Set conceptors in evaluation mode and generate a sample
for i in range(4):
    # Evaluation mode
    conceptors[i].eval(True)

    # Set it as current conceptor
    conceptor_net.set_conceptor(conceptors[i])

    # Generate sample
    pattern_sample = conceptor_net(torch.zeros(1, conceptor_test_length, 1, dtype=dtype))
    plt.plot(pattern_sample[0])
    plt.show()
    # Find the best matching position with cubic interpolation

    # Save position for plotting
# end for

# Create plot for each pattern
for i in range(4):
    pass
# end for


