# -*- coding: utf-8 -*-
#
# File : examples/conceptors/subspace_demo.py
# Description : Conceptor first subspace demo
# Date : 5th of December, 2019
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

# Plots
n_plots = 9

# Morphing
min_mu = -0.5
max_mu = 1.5
morphing_range = [0.0, 1.0]
morphing_length = 30
morphing_washout = 190
morphing_plot_length = 15
morphing_plot_points = 15

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
    w_generator = mg.matrix_factory.get_generator(
        "matlab",
        file_name=args.w,
        entity_name=args.w_name,
        scale=spectral_radius
    )
else:
    # Generate internal weights
    w_generator = mg.matrix_factory.get_generator(
        "normal",
        mean=0.0,
        std=1.0,
        connectivity=connectivity,
        spectral_radius=spectral_radius
    )
# end if

# Load Win from matlab file or init randomly
if args.win != "":
    # Load internal weights
    win_generator = mg.matrix_factory.get_generator(
        "matlab",
        file_name=args.win,
        entity_name=args.win_name,
        scale=input_scaling
    )
else:
    # Generate Win
    win_generator = mg.matrix_factory.get_generator(
        "normal",
        mean=0.0,
        std=1.0,
        connectivity=1.0,
        scale=input_scaling
    )
# end if

# Load Wbias from matlab from or init randomly
if args.wbias != "":
    wbias_generator = mg.matrix_factory.get_generator(
        "matlab",
        file_name=args.wbias,
        entity_name=args.wbias_name,
        shape=reservoir_size,
        scale=bias_scaling
    )
else:
    wbias_generator = mg.matrix_factory.get_generator(
        "normal",
        mean=0.0,
        std=1.0,
        connectivity=1.0,
        scale=bias_scaling
    )
# end if

# Load x0 from matlab from or init randomly
if args.x0 != "":
    print("x0!")
    x0_generator = mg.matrix_factory.get_generator("matlab", file_name=args.x0, entity_name=args.x0_name, shape=reservoir_size)
else:
    x0_generator = mg.matrix_factory.get_generator("normal", mean=0.0, std=1.0, connectivity=1.0)
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
conceptors = ecnc.ConceptorSet(input_dim=reservoir_size, dtype=dtype)

# Create four conceptors, one for each pattern
conceptors.add(0, ecnc.Conceptor(input_dim=reservoir_size, aperture=alpha, dtype=dtype))
conceptors.add(1, ecnc.Conceptor(input_dim=reservoir_size, aperture=alpha, dtype=dtype))
conceptors.add(2, ecnc.Conceptor(input_dim=reservoir_size, aperture=alpha, dtype=dtype))
conceptors.add(3, ecnc.Conceptor(input_dim=reservoir_size, aperture=alpha, dtype=dtype))

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
    fill_left=True,
    dtype=dtype
)

# We create an outside observer to plot
# internal states and SVD afterwards
observer = ecvs.NodeObserver(conceptor_net.cell, initial_state='init')

# Xold and Y collectors
Xold_collector = torch.empty(4 * learn_length, reservoir_size, dtype=dtype)
Y_collector = torch.empty(4 * learn_length, reservoir_size, dtype=dtype)
P_collector = torch.empty(4, signal_plot_length, dtype=dtype)
last_X = torch.empty(4, reservoir_size, dtype=dtype)

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
    last_X[i] = X[0, -1]
# end for

# Observer set as inactive, it will stop observing
# reservoir states and inputs.
observer.set_active(False)

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
# generated = conceptor_net(torch.zeros(1, conceptor_test_length, 1, dtype=dtype))

# Plot the generated signal
# plt.title("Messy output after loading W")
# plt.plot(generated[0], color='r', linewidth=2)
# plt.show()

# Run loaded reservoir to observe a messy output. Do this with starting
# from four states originally obtained in the four driving conditions
# initialize network state.

# Figure (square size)
plt.figure(figsize=(12, 8))

# For each pattern
for p in range(4):
    # Set hidden state
    conceptor_net.cell.set_hidden(last_X[p])

    # Run trained ESN with empty inputs (no conceptor learning)
    generated = conceptor_net(torch.zeros(1, conceptor_test_length, 1, dtype=dtype), reset_state=False)

    # Select subplot and plot the messy output
    plt.subplot(4, 1, p + 1)
    plt.plot(generated[0].numpy(), color='b')
# end for

# Show figure
plt.show()

# Conceptors ON
conceptor_net.conceptor_active(True)

# Train conceptors (Compute C from R)
conceptors.finalize()

# Corresponding mixture vectors
mixture_vectors = torch.empty((n_plots, n_plots, 1, 4))

# Rows and columns
row_mus = torch.linspace(min_mu, max_mu, n_plots)
col_mus = torch.linspace(min_mu, max_mu, n_plots)

# Compute mixture vectors
for i in range(n_plots):
    for j in range(n_plots):
        # The first two entries in mixture_vectors relate to the first two patterns,
        # the second two entries to the last two patterns.
        mixture_vectors[i, j, 0, :2] = row_mus[i] * torch.Tensor([1.0 - col_mus[j], col_mus[j]])
        mixture_vectors[i, j, 0, 2:] = (1.0 - row_mus[i]) * torch.Tensor([1.0 - col_mus[j], col_mus[j]])
    # end for
# end for

# Morphing washout
conceptor_net.washout = morphing_washout

# Output for each mixture
plots = torch.empty((n_plots, n_plots, morphing_length))

# Randomly generated initial state (x0)
x0 = x0_generator.generate(size=reservoir_size, dtype=dtype)

# For each morphing
for i in range(n_plots):
    for j in range(n_plots):
        # Mixture vector
        mixture_vector = mixture_vectors[i, j]

        # Randomly generated initial state (x0)
        conceptor_net.cell.set_hidden(x0)

        # Generate sample
        generated_sample = conceptor_net(
            torch.zeros(1, morphing_length + morphing_washout, 1, dtype=dtype),
            reset_state=False,
            morphing_vectors=mixture_vector
        )
        if i == 0 and j == 0:
            print("{} {}".format(i, j))
            print(generated_sample[0])
        elif i == 8 and j == 0:
            print("{} {}".format(i, j))
            print(generated_sample[0])
        elif i == 0 and j == 8:
            print("{} {}".format(i, j))
            print(generated_sample[0])
        elif i == 8 and j == 8:
            print("{} {}".format(i, j))
            print(generated_sample[0])
        # end if
        # Save outputs
        plots[i, j] = generated_sample[0, :, 0]
    # end for
# end for

# Figure (square size)
plt.figure(figsize=(18, 18))

# Panel index
panel_index = 1

# For each morphing
for i in range(n_plots):
    for j in range(n_plots):
        # Subplot
        plt.subplot(n_plots, n_plots, panel_index)

        # Morphing data
        thisdata = plots[i, j, :morphing_plot_points]

        # Title
        plt.title(
            "{},{},{},{}".format(
                mixture_vectors[i, j, 0, 0],
                mixture_vectors[i, j, 0, 1],
                mixture_vectors[i, j, 0, 2],
                mixture_vectors[i, j, 0, 3]
            ),
            fontsize=8
        )

        # Y label
        if j == 0:
            plt.ylabel(row_mus[i].item())
        # end if

        # X label
        if i == n_plots - 1:
            plt.xlabel(col_mus[j].item())
        # end if

        # Limits and ticks
        plt.xticks([])
        plt.yticks([])
        plt.ylim([-1, 1])

        # Original timeseries
        n = (n_plots - 5.0) / 4.0
        if (i == n + 1 and j == n + 1) or (i == n + 1 and j == 3 * n + 3) or (i == 3 * n + 3 and j == n + 1) or (i == 3 * n + 3 and j == 3 * n + 3):
            plt.plot(thisdata, 'r', linewidth=3)
        else:
            plt.plot(thisdata, 'g', linewidth=3)
        # end

        # Next panel
        panel_index += 1
    # end
# end

# Show
plt.show()

