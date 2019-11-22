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
import echotorch.utils.visualisation as ecvs
from echotorch.datasets import DatasetComposer
from echotorch.nn.Node import Node
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
from scipy.interpolate import interp1d
import numpy.linalg as lin

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
interpolation_rate = 20

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
    learning_algo='inv',
    w_generator=w_generator,
    win_generator=win_generator,
    wbias_generator=wbias_generator,
    input_scaling=input_scaling,
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
    esn_cell=spesn.cell,
    dtype=dtype
)

# We create an outside observer to plot
# internal states and SVD afterwards
observer = ecvs.ESNCellObserver(spesn.cell)

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
P_collector = torch.empty(4, signal_plot_length, dtype=dtype)

# Create four conceptors, one for each pattern
conceptors = [
    ecnc.Conceptor(input_dim=reservoir_size, aperture=alpha, dtype=dtype),
    ecnc.Conceptor(input_dim=reservoir_size, aperture=alpha, dtype=dtype),
    ecnc.Conceptor(input_dim=reservoir_size, aperture=alpha, dtype=dtype),
    ecnc.Conceptor(input_dim=reservoir_size, aperture=alpha, dtype=dtype)
]

# Conceptors ON
conceptor_net.conceptor_active(True)

# Go through dataset
for i, data in enumerate(patterns_loader):
    # Inputs and labels
    inputs, outputs, labels = data

    # To Variable
    if dtype == torch.float64:
        inputs, outputs = Variable(inputs.double()), Variable(outputs.double())
    # end if

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
    P_collector[i] = inputs[0, washout_length:washout_length+signal_plot_length, 0]
# end for

# Learn internal weights
conceptor_net.finalize()

# Predicted by W
predY = torch.mm(conceptor_net.cell.w, Xold_collector.t()).t()

# Compute NRMSE
training_NRMSE = echotorch.utils.nrmse(predY, Y_collector)
print("Training NRMSE : {}".format(training_NRMSE))

# Conceptors OFF
conceptor_net.conceptor_active(False)

# No washout this time
conceptor_net.washout = 0

# Run trained ESN with empty inputs (no conceptor learning)
generated = conceptor_net(torch.zeros(1, conceptor_test_length, 1, dtype=dtype))

# Plot the generated signal
plt.title("Messy output after loading W")
plt.plot(generated[0], color='r', linewidth=2)
plt.show()

# Conceptors ON
conceptor_net.conceptor_active(True)

# Save each generated pattern for display
generated_samples = torch.zeros(4, conceptor_test_length)

# NRMSE between original and aligned pattern
NRMSEs_aligned = torch.zeros(4)

# Figure (square size)
plt.figure(figsize=(12, 8))

# Set conceptors in evaluation mode and generate a sample
for i in range(4):
    # Train conceptors
    conceptors[i].finalize()

    # Set it as current conceptor
    conceptor_net.set_conceptor(conceptors[i])

    # Generate sample
    generated_sample = conceptor_net(torch.zeros(1, conceptor_test_length, 1, dtype=dtype))

    # Find best phase shift
    generated_sample_aligned, _, NRMSE_aligned = echotorch.utils.pattern_interpolation(P_collector[i], generated_sample[0], interpolation_rate)

    # Plot 1 : original pattern and recreated pattern
    plt.subplot(4, 4, i * 4 + 1)
    plt.plot(generated_sample_aligned, color='r', linewidth=5)
    plt.plot(P_collector[i], color='b', linewidth=1.5)

    # Title
    if i == 0:
        plt.title('p vs y')
    # end if

    # X labels
    if i == 3:
        plt.xticks([0, 10, 20])
    else:
        plt.xticks([])
    # end if

    # Y limits
    plt.ylim([-1, 1])
    plt.yticks([-1, 0, 1])

    # Plot 2 : neurons
    plt.subplot(4, 4, i * 4 + 2)
    observer.plot_neurons(
        sample_id=i,
        idxs=[0, 1, 2],
        length=signal_plot_length,
        color='g',
        linewidth=1.5,
        show_title=(i==0),
        title="Two neurons",
        xticks=[0, 10, 20] if i == 3 else None,
        ylim=[-1, 1],
        yticks=[-1, 0, 1]
    )

    # Plot 3 : Log10 of singular values (PC energy)
    plt.subplot(4, 4, i * 4 + 3)
    observer.plot_state_singular_values(
        sample_id=i,
        color='r',
        linewidth=2,
        show_title=(i == 0),
        title="Log10 PC Energy",
        xticks=[0, 50, 100] if i == 3 else None,
        ylim=[-20, 10],
        log10=True
    )

    # Plot 4 : Learning PC energy
    plt.subplot(4, 4, i * 4 + 4)
    observer.plot_state_singular_values(
        sample_id=i,
        color='r',
        length=10,
        linewidth=2,
        show_title=(i == 0),
        title="Leading PC energy",
        ylim=[0, 40.0]
    )

    # Save NRMSE
    NRMSEs_aligned[i] = NRMSE_aligned
# end for

# Show
plt.show()

# Show NRMSE
print("NRMSEs aligned : {}".format(torch.mean(NRMSEs_aligned)))

# Create plot for each pattern
for i in range(4):
    pass
# end for


