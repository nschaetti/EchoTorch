# -*- coding: utf-8 -*-
#
# File : examples/conceptors/morphing_sines.py
# Description : Morphing sines with Conceptors
# Date : 6th of December, 2019
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
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
from scipy.interpolate import interp1d
from scipy import signal

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

# Patterns
sine1_period_length = 8.8342522
sine2_period_length = 9.8342522

# Sequence lengths
washout_length = 500
learn_length = 1000
signal_plot_length = 20
conceptor_test_length = 200
singular_plot_length = 50

# Regularization
ridge_param_wstar = 0.01
ridge_param_wout = 0.01

# Plots
n_plots = 9

# Morphing
morphing_range = [-2, 3]
morphing_length = 200
morphing_washout = 500
morphing_pre_record_length = 50
morphing_delay_time = 500
morphing_delay_plot_points = 25
morphing_total_length = morphing_washout + morphing_pre_record_length * 2 + morphing_length
morphing_n_top_plots = 8

# Aperture
alpha = 10

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

# Load x0 from matlab from or init randomly
if args.x0 != "":
    print("x0!")
    x0_generator = mg.matrix_factory.get_generator("matlab", file_name=args.x0, entity_name=args.x0_name, shape=reservoir_size)
else:
    x0_generator = mg.matrix_factory.get_generator("normal", mean=0.0, std=1.0, connectivity=1.0)
# end if

# Four pattern (two sine, two periodic)
pattern1_training = etds.SinusoidalTimeseries(sample_len=washout_length + learn_length, n_samples=1, a=1,
    period=sine1_period_length,
    dtype=dtype
)
pattern2_training = etds.SinusoidalTimeseries(sample_len=washout_length + learn_length, n_samples=1, a=1,
    period=sine2_period_length,
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
    input_scaling=1.0,
    ridge_param=ridge_param_wout,
    w_ridge_param=ridge_param_wstar,
    washout=washout_length,
    dtype=dtype
)

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
    esn_cell=spesn.cell,
    conceptor=conceptors,
    dtype=dtype
)

# We create an outside observer to plot
# internal states and SVD afterwards
observer = ecvs.NodeObserver(spesn.cell, initial_state='init')

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

# Train conceptors (Compute C from R)
conceptors.finalize()

# Morphing values
morphing_scales = np.linspace(morphing_range[0], morphing_range[1], morphing_length + 1)

# Morphing outputs
morphing_output = torch.zeros(morphing_total_length)

# Initialize morphing vectors
morphing_vectors = torch.zeros(1, morphing_total_length, 4)

# Morphing vectors for the washout and pre-morphing periods
for t in range(morphing_washout + morphing_pre_record_length):
    morphing_vectors[0, t, 0] = 1.0 - morphing_scales[0]
    morphing_vectors[0, t, 1] = morphing_scales[0]
# end for

# Morphing vectors for the morphing period
for t in range(morphing_length):
    add_time = morphing_washout + morphing_pre_record_length
    morphing_vectors[0, t + add_time, 0] = 1.0 - morphing_scales[t]
    morphing_vectors[0, t + add_time, 1] = morphing_scales[t]
# end for

# Morphing vectors for the post-morphing period
for t in range(morphing_pre_record_length):
    add_time = morphing_washout + morphing_pre_record_length + morphing_length
    morphing_vectors[0, t + add_time, 0] = 1.0 - morphing_scales[-1]
    morphing_vectors[0, t + add_time, 1] = morphing_scales[-1]
# end for

# Length to plot
morphing_plot_length = morphing_length + morphing_pre_record_length * 2

# Randomly generated initial state (x0)
x0 = x0_generator.generate(size=reservoir_size, dtype=dtype)

# Washout
conceptor_net.washout = morphing_washout

# Generate sample
morphed_outputs = conceptor_net(
    torch.zeros(1, morphing_total_length, 1, dtype=dtype),
    reset_state=False,
    morphing_vectors=morphing_vectors
)

plt.plot(morphed_outputs[0, :, 0].numpy(), color='r')
plt.show()

# We compute a quadratic interpolation of the output
# morphed signal.
interpolation_increment = 0.1
interpolation_points = np.arange(1, morphing_plot_length, interpolation_increment) - 1.0
interpolation_length = interpolation_points.shape[0]
morphed_outputs_interpolated = interp1d(
    np.arange(morphing_plot_length),
    morphed_outputs[0, :, 0].numpy(),
    kind='quadratic'
)(interpolation_points)

plt.plot(morphed_outputs_interpolated, color='r')
plt.show()

# We compute the length of each period (number of timesteps)
# by counting the gap between each point where the signal
# cross the x-axis from below.
x_crossing_discounts = np.zeros(interpolation_length)
old_val = 1
counter = 0
for i in range(interpolation_length-1):
    # Cross x-axis from below
    if morphed_outputs_interpolated[i] < 0 and morphed_outputs_interpolated[i+1] > 0:
        counter += 1
        x_crossing_discounts[i] = counter
        old_val = counter
        counter = 0
    else:
        x_crossing_discounts[i] = old_val
        counter += 1
    # end if
# end for

plt.plot(x_crossing_discounts, color='pink')
plt.show()

# We get periods length back to the sampling rate
# More remove first 20, and the last was which are
# not accurate enough.
interpolation_steps = int(1.0 / interpolation_increment)
x_crossing_discounts = x_crossing_discounts[range(interpolation_steps - 1, interpolation_length, interpolation_steps)]
x_crossing_discounts *= interpolation_increment
x_crossing_discounts[:20] = np.ones(20) * x_crossing_discounts[19]
x_crossing_discounts[-1] = x_crossing_discounts[-2]

plt.plot(x_crossing_discounts, color='pink')
plt.show()

# We compute expected period lengths at the start of the morphing
# and at the end
sines_period_diffs = sine2_period_length - sine1_period_length
period_length_start = sine1_period_length + morphing_range[0] * sines_period_diffs
period_length_end = sine1_period_length + morphing_range[1] * sines_period_diffs

# We compute a reference to display the expected period length on future plots
period_lengths_plotting = np.zeros(morphing_plot_length)
period_lengths_plotting[:morphing_pre_record_length] = period_length_start * np.ones(morphing_pre_record_length)
period_lengths_plotting[morphing_pre_record_length:morphing_pre_record_length+morphing_length] = \
    np.linspace(
        period_length_start,
        period_length_end, morphing_length
    )
period_lengths_plotting[morphing_pre_record_length+morphing_length:] = \
    period_length_end * np.ones(morphing_pre_record_length)

plt.plot(period_lengths_plotting, color='b')
plt.show()

# Two points to show the period lengths of the two patterns used in the training phase
morphing_point_sine1 = morphing_pre_record_length + morphing_length * \
                       -morphing_range[0] / float(morphing_range[1] - morphing_range[0])
morphing_point_sine2 = morphing_pre_record_length + morphing_length * \
                       -(morphing_range[0] - 1) / float(morphing_range[1] - morphing_range[0])

# Morphing values for each plots
morphing_delay_ms = np.linspace(morphing_range[0], morphing_range[1], morphing_n_top_plots)
morphing_delay_data = torch.zeros(morphing_n_top_plots, morphing_delay_time)

# Random initial state
x0 = torch.randn(reservoir_size)

# Washout
conceptor_net.washout = morphing_washout

# For each morphing value
for i in range(morphing_n_top_plots):
    # Morphing vector
    morphing_vector = torch.DoubleTensor([1.0 - morphing_delay_ms[i], morphing_delay_ms[i], 0.0, 0.0])
    morphing_vector = morphing_vector.reshape(1, 4)

    # Generate delay output
    morphed_outputs = conceptor_net(
        torch.zeros(1, morphing_delay_time + morphing_washout, 1, dtype=dtype),
        reset_state=False,
        morphing_vectors=morphing_vector
    )

    morphing_delay_data[i] = morphed_outputs[0, :, 0]
# end for

# Top plots finger points
morphing_top_plots_points = morphing_pre_record_length + np.arange(0, morphing_n_top_plots) * \
                            morphing_length / (morphing_n_top_plots - 1)

# New figure
plt.figure(figsize=(16, 2))

# For each top plots
plot_index = 0
for i in range(morphing_n_top_plots):
    # Subplot
    plt.subplot(1, morphing_n_top_plots, plot_index + 1)
    plot_index += 1

    # Plot data
    plt.plot(morphing_delay_data[i, :-2], morphing_delay_data[i, 1:-1], 'o-', color='red', markersize=1, linewidth=1)

    # Plot points
    plot_data = morphing_delay_data[i, np.arange(0, morphing_delay_plot_points + 1)]
    plt.plot(plot_data[:-2], plot_data[1:-1], 'o', color='blue', markersize=8)

    # Limits
    plt.xlim([-1.4, 1.4])
    plt.ylim([-1.4, 1.4])
    plt.xticks([])
    plt.yticks([])
# end for

# Show
plt.show()

# Second figure
plt.figure(figsize=(14, 6))

# Subplot 1
plt.subplot(2, 1, 1)
plt.plot(morphed_outputs[0, :, 0].numpy(), '-', color='red', linewidth=2)
plt.plot([morphing_point_sine1, morphing_point_sine2], [-1, -1], 'o', color='green', markersize=15)
plt.plot(morphing_top_plots_points, 1.1 * np.ones(morphing_n_top_plots), 'x', color='blue', markersize=10)
plt.grid(color='grey', linestyle='--', linewidth=1, axis='x')
plt.xlim([0, morphing_plot_length])
plt.ylim([-1.3, 1.3])

# Subplot 2
plt.subplot(2, 1, 2)
plt.plot(x_crossing_discounts, linewidth=6, color='magenta')
plt.plot(period_lengths_plotting, color='blue')
plt.plot([morphing_point_sine1, morphing_point_sine2], np.array([7, 7]), 'o', color='green', markersize=15)
plt.xlim([0, morphing_plot_length])
plt.ylim([6.5, 12.5])
plt.xticks([])

# Show
plt.show()
