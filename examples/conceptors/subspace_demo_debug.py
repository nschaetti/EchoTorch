# -*- coding: utf-8 -*-
#
# File : examples/conceptors/subspace_demo_debug.py
# Description : First subspace demo with conceptors (debug mode).
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
from examples.conceptors.patterns.periodic_patterns import pattern_library

# Debug ?
debug = True
if debug:
    torch.set_printoptions(precision=16)
    debug_mode = Node.DEBUG_OUTPUT
    precision = 0.0001
else:
    debug_mode = Node.NO_DEBUG
# end if

# Random numb. init
torch.random.manual_seed(1)
np.random.seed(1)

# Type parameters
dtype=torch.float64

# Reservoir parameters
reservoir_size = 100
spectral_radius = 1.5
bias_scaling = 0.2
connectivity = 10.0 / reservoir_size

# Input parameters
input_scaling = 1.5

# Learning parameters
washout_length = 500
learn_length = 1000
loading_method = ecnc.SPESNCell.W_LOADING

# Testing parameters
conceptor_test_length = 200
interpolation_rate = 20

# Plotting parameters
signal_plot_length = 20
singular_plot_length = 50

# Regularization
ridge_param_wstar = 0.0001
ridge_param_wout = 0.01

# Aperture
alpha = 10
gamma = 10.0

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
    w_generator = mg.matrix_factory.get_generator("matlab", file_name=args.w, entity_name=args.w_name, scale=spectral_radius)
else:
    # Generate internal weights
    w_generator = mg.matrix_factory.get_generator("normal", mean=0.0, std=1.0, connectivity=connectivity)
# end if

# Load Win from matlab file or init randomly
if args.win != "":
    # Load internal weights
    win_generator = mg.matrix_factory.get_generator("matlab", file_name=args.win, entity_name=args.win_name, scale=input_scaling)
else:
    # Generate Win
    win_generator = mg.matrix_factory.get_generator("normal", mean=0.0, std=1.0, connectivity=1.0)
# end if

# Load Wbias from matlab from or init randomly
if args.wbias != "":
    wbias_generator = mg.matrix_factory.get_generator("matlab", file_name=args.wbias, entity_name=args.wbias_name, shape=reservoir_size, scale=bias_scaling)
else:
    wbias_generator = mg.matrix_factory.get_generator("normal", mean=0.0, std=1.0, connectivity=1.0)
# end if


# First sine periodic pattern
pattern1_training = pattern_library(pattern_id=0, washout_length=washout_length, learn_length=learn_length)

# Second sine periodic pattern
pattern2_training = pattern_library(pattern_id=1, washout_length=washout_length, learn_length=learn_length)

# First 5-periodic pattern
pattern3_training = pattern_library(pattern_id=4, washout_length=washout_length, learn_length=learn_length)

# Second 5-periodic pattern
pattern4_training = pattern_library(pattern_id=5, washout_length=washout_length, learn_length=learn_length)

# Composer
dataset_training = DatasetComposer([pattern1_training, pattern2_training, pattern3_training, pattern4_training])

# Data loader
patterns_loader = DataLoader(dataset_training, batch_size=1, shuffle=False, num_workers=1)

# Create a set of conceptors
conceptors = ecnc.ConceptorSet(input_dim=reservoir_size)

# Create four conceptors, one for each pattern
for c_i in range(4):
    conceptors.add(c_i, ecnc.Conceptor(
        input_dim=reservoir_size,
        aperture=alpha,
        debug=debug_mode,
        dtype=dtype
    ))
# end for

# Create a conceptor network using
# the self-predicting ESN which
# will learn four conceptors.
conceptor_net = ecnc.ConceptorNet(
    input_dim=1,
    hidden_dim=reservoir_size,
    output_dim=1,
    conceptor=conceptors,
    learning_algo='inv',
    w_generator=w_generator,
    win_generator=win_generator,
    wbias_generator=wbias_generator,
    input_scaling=1.0,
    ridge_param=ridge_param_wout,
    w_ridge_param=ridge_param_wstar,
    washout=washout_length,
    loading_method=loading_method,
    debug=debug_mode,
    dtype=dtype
)

# We create an outside observer to plot
# internal states and SVD afterwards
observer = ecvs.NodeObserver(conceptor_net.cell, initial_state='init')

# If in debug mode
if debug_mode > Node.NO_DEBUG:
    # Load sample matrices
    for i in range(4):
        # Input patterns
        conceptor_net.cell.debug_point(
            "u{}".format(i),
            torch.reshape(torch.from_numpy(np.load("data/debug/subspace_demo/u{}.npy".format(i))), shape=(-1, 1)),
            precision
        )

        # States
        conceptor_net.cell.debug_point(
            "X{}".format(i),
            torch.from_numpy(np.load("data/debug/subspace_demo/X{}.npy".format(i))),
            precision
        )

        # Targets
        if loading_method == ecnc.SPESNCell.W_LOADING:
            conceptor_net.cell.debug_point(
                "Y{}".format(i),
                torch.from_numpy(np.load("data/debug/subspace_demo/Y{}.npy".format(i))),
                precision
            )
        # end if

        # Xold
        conceptor_net.cell.debug_point(
            "Xold{}".format(i),
            torch.from_numpy(np.load("data/debug/subspace_demo/Xold{}.npy".format(i))),
            precision
        )

        # Conceptor
        conceptors[i].debug_point(
            "C",
            torch.from_numpy(np.load("data/debug/subspace_demo/C{}.npy".format(i))),
            precision
        )
    # end for

    # Load debug W, xTx, xTy
    conceptor_net.cell.debug_point("Wstar", torch.from_numpy(np.load("data/debug/subspace_demo/Wstar.npy", allow_pickle=True)), precision)
    conceptor_net.cell.debug_point("Win", torch.from_numpy(np.load("data/debug/subspace_demo/Win.npy")), precision)
    conceptor_net.cell.debug_point("Wbias", torch.from_numpy(np.load("data/debug/subspace_demo/Wbias.npy")), precision)
    conceptor_net.cell.debug_point("xTx", torch.from_numpy(np.load("data/debug/subspace_demo/xTx.npy")), precision)
    conceptor_net.cell.debug_point("w_ridge_param", 0.0001, precision)
    conceptor_net.cell.debug_point("ridge_xTx", torch.from_numpy(np.load("data/debug/subspace_demo/ridge_xTx.npy")), precision)
    conceptor_net.cell.debug_point("inv_xTx", torch.from_numpy(np.load("data/debug/subspace_demo/inv_xTx.npy")), precision)
    conceptor_net.cell.debug_point("w", torch.from_numpy(np.load("data/debug/subspace_demo/W.npy")), precision)

    # Debug not related to inputs recreation
    if loading_method != ecnc.SPESNCell.INPUTS_RECREATION:
        conceptor_net.cell.debug_point("xTy", torch.from_numpy(np.load("data/debug/subspace_demo/xTy.npy")), precision)
    # end if
# end if

# Xold and Y collectors
Xold_collector = torch.empty(4 * learn_length, reservoir_size, dtype=dtype)
Y_collector = torch.empty(4 * learn_length, reservoir_size, dtype=dtype)
P_collector = torch.empty(4, signal_plot_length, dtype=dtype)

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

    # Set conceptor to use
    conceptors.set(i)

    # Set state of the observer
    observer.set_state("pattern{}".format(i))

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

# Observer set as inactive, it will stop observing
# reservoir states and inputs.
observer.set_active(False)

# Learn internal weights
conceptor_net.finalize()
# print(conceptor_net.cell.D)

# Predicted by W
predY = torch.mm(conceptor_net.cell.w, Xold_collector.t()).t()

# Compute NRMSE
if loading_method == ecnc.SPESNCell.W_LOADING:
    training_NRMSE = echotorch.utils.nrmse(predY, Y_collector)
    print("Training NRMSE : {}".format(training_NRMSE))
# end if

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

# Train conceptors (Compute C from R)
conceptors.finalize()

# Figure (square size)
plt.figure(figsize=(12, 8))

# Set conceptors in evaluation mode and generate a sample
for i in range(4):
    # Set it as current conceptor
    conceptors.set(i)

    # Randomly generated initial state (x0)
    conceptor_net.cell.set_hidden(0.5 * torch.randn(reservoir_size, dtype=dtype))

    # Generate sample
    generated_sample = conceptor_net(torch.zeros(1, conceptor_test_length, 1, dtype=dtype), reset_state=False)

    # Find best phase shift
    generated_sample_aligned, _, NRMSE_aligned = echotorch.utils.pattern_interpolation(P_collector[i], generated_sample[0], interpolation_rate)

    #
    # Plot 1 : original pattern and recreated pattern
    #
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

    # We use StateVisualiser to plot neural activities of
    # two reservoir units, the log10 of singular values
    # of reservoir states, and their 10 leading SV.
    state_visualiser = ecvs.StateVisualiser(observer=observer)

    #
    # Plot 2 : neurons
    #
    plt.subplot(4, 4, i * 4 + 2)

    # Plot neurons
    state_visualiser.plot_neurons(
        point_name='X',
        states="pattern{}".format(i),
        idxs=None,
        neuron_idxs=[0, 1, 2],
        length=signal_plot_length,
        colors=['b', 'orange', 'g'],
        linewidth=1.5,
        show_title=(i==0),
        title="Two neurons",
        xticks=[0, 10, 20] if i == 3 else None,
        yticks=[-1, 0, 1],
        ylim=[-1, 1]
    )

    #
    # Plot 3 : Log10 of singular values (PC energy)
    #
    plt.subplot(4, 4, i * 4 + 3)

    # Plot log10 of SV
    state_visualiser.plot_singular_values(
        point_name='X',
        states="pattern{}".format(i),
        idxs=None,
        color='r',
        linewidth=2,
        show_title=(i == 0),
        title="Log 10 PC Energy",
        xticks=[0, 50, 100] if i == 3 else None,
        ylim=[-20, 10],
        log10=True
    )

    # Plot 4 : Learning PC energy
    plt.subplot(4, 4, i * 4 + 4)

    # Plot learning SV
    state_visualiser.plot_singular_values(
        point_name='X',
        states="pattern{}".format(i),
        idxs=None,
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

# Plot R similarity matrix
ecvs.show_similarity_matrix(
    sim_matrix=conceptors.similarity_matrix(based_on='R'),
    title="R base similarities"
)

# Plot conceptors similarity matrix at aperture = 10.0
ecvs.show_similarity_matrix(
    sim_matrix=conceptors.similarity_matrix(),
    title="C based similarities, aperture = {}".format(alpha)
)

# Take conceptor for pattern 1 (sine) and pattern 3 (periodic)
Cs = conceptors[0]
Cp = conceptors[2]

# Divide aperture by 10 (to get aperture = 1.0)
Cs.PHI(1.0 / 10.0)
Cp.PHI(1.0 / 10.0)

# Figure with two plots
fig = plt.figure(figsize=(14, 6))

# Plots color
colors = ['b', 'orange', 'g', 'red', 'purple']

# Plot labels
plot_labels = ["a = 1.0", "a = 10.0", "a = 100.0", "a = 1000.0", "a = 10000.0"]

# Select first figure
plt.subplot(1, 2, 1)

# For each aperture (1.0, 10.0, 100.0, 1000.0, 10000.0)
for i in range(5):
    # Plot conceptor's singular values.
    plt.plot(Cs.SV.numpy(), color=colors[i], label=plot_labels[i])

    # Title
    if i == 0:
        plt.title("Sine singular values")
    # end if

    # Legend
    plt.legend(loc='best', ncol=1)

    # Multiply aperture by 10.0
    Cs.PHI(10.0)
# end for

# Select second figure
plt.subplot(1, 2, 2)

# For each aperture (1.0, 10.0, 100.0, 1000.0, 10000.0)
for i in range(5):
    # Plot conceptor's singular values.
    plt.plot(Cp.SV.numpy(), color=colors[i], label=plot_labels[i])

    # Title
    if i == 0:
        plt.title("Periodic singular values")
    # end if

    # Legend
    plt.legend(loc='best', ncol=1)

    # Multiply aperture by 10.0
    Cp.PHI(10.0)
# end for

# Show
plt.show()
