# -*- coding: utf-8 -*-
#
# File : examples/conceptors/memory_management.py
# Description : Conceptor memory management demo
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
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Random numb. init
torch.random.manual_seed(1)
np.random.seed(1)

# region PARAMS

# Type params
dtype=torch.float64

# Reservoir params
reservoir_size = 100
spectral_radius = 1.5
bias_scaling = 0.2
connectivity = 10.0 / reservoir_size

# Inputs params
input_scaling = 1.5

# Sequence lengths
washout_length = 100
learn_length = 100

# Test params
interpolation_rate = 20
conceptor_test_length = 200
conceptor_test_washout = 200

# Regularization params
ridge_param_wout = 0.01

# Plot params
signal_plot_length = 20
n_plot_singular_values = 100

# Aperture
aperture = 1000

# Pattern params
n_patterns = 16

# endregion PARAMS

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

# region MATRICES_INIT

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
    x0_generator = mg.matrix_factory.get_generator(
        "matlab",
        file_name=args.x0,
        entity_name=args.x0_name,
        shape=reservoir_size
    )
else:
    x0_generator = mg.matrix_factory.get_generator(
        "normal",
        mean=0.0,
        std=1.0,
        connectivity=1.0
    )
# end if

# endregion MATRICES_INIT

# region CREATE_PATTERNS

# Pattern 1 (sine p=10)
pattern1_training = etds.SinusoidalTimeseries(sample_len=washout_length + learn_length, n_samples=1, a=1,
    period=10,
    dtype=dtype
)

# Pattern 2 (sine p=15)
pattern2_training = etds.SinusoidalTimeseries(sample_len=washout_length + learn_length, n_samples=1, a=1,
    period=15,
    dtype=dtype
)

# Pattern 3 (periodic 4)
pattern3_training = etds.PeriodicSignalDataset(sample_len=washout_length + learn_length, n_samples=1,
    period=[-0.4564, 0.6712, -2.3953, -2.1594],
    dtype=dtype
)

# Pattern 4 (periodic 6)
pattern4_training = etds.PeriodicSignalDataset(sample_len=washout_length + learn_length, n_samples=1,
    period=[0.5329, 0.9621, 0.1845, 0.5099, 0.3438, 0.7697],
    dtype=dtype
)

# Pattern 5 (periodic 7)
pattern5_training = etds.PeriodicSignalDataset(sample_len=washout_length + learn_length, n_samples=1,
    period=[0.8029, 0.4246, 0.2041, 0.0671, 0.1986, 0.2724, 0.5988],
    dtype=dtype
)

# Pattern 6 (sine p=12)
pattern6_training = etds.SinusoidalTimeseries(sample_len=washout_length + learn_length, n_samples=1, a=1,
    period=12,
    dtype=dtype
)

# Pattern 7 (sine p=5)
pattern7_training = etds.SinusoidalTimeseries(sample_len=washout_length + learn_length, n_samples=1, a=1,
    period=5,
    dtype=dtype
)

# Pattern 8 (sine p=6)
pattern8_training = etds.SinusoidalTimeseries(sample_len=washout_length + learn_length, n_samples=1, a=1,
    period=6,
    dtype=dtype
)

# Pattern 9 (periodic 8)
pattern9_training = etds.PeriodicSignalDataset(sample_len=washout_length + learn_length, n_samples=1,
    period=[0.8731, 0.1282, 0.9582, 0.6832, 0.7420, 0.9829, 0.4161, 0.5316],
    dtype=dtype
)

# Pattern 10 (periodic 7)
pattern10_training = etds.PeriodicSignalDataset(sample_len=washout_length + learn_length, n_samples=1,
    period=[0.6792, 0.5129, 0.2991, 0.1054, 0.2849, 0.7689, 0.6408],
    dtype=dtype
)

# Pattern 11 (periodic 3)
pattern11_training = etds.PeriodicSignalDataset(sample_len=washout_length + learn_length, n_samples=1,
    period=[1.4101, -0.0992, -0.0902],
    dtype=dtype
)

# Pattern 12 (sine p=6)
pattern12_training = etds.SinusoidalTimeseries(sample_len=washout_length + learn_length, n_samples=1, a=1,
    period=11,
    dtype=dtype
)

# Pattern 13 (periodic 5)
pattern13_training = etds.PeriodicSignalDataset(sample_len=washout_length + learn_length, n_samples=1,
    period=[0.3419, 1.1282, 0.3107, -0.8949, -0.7266],
    dtype=dtype
)


# Composer
dataset_training = DatasetComposer([
    pattern1_training, pattern2_training, pattern3_training, pattern4_training, pattern1_training, pattern2_training,
    pattern3_training, pattern5_training, pattern6_training, pattern7_training, pattern8_training, pattern9_training, pattern10_training, pattern11_training,
    pattern12_training, pattern13_training
])

# Data loader
patterns_loader = DataLoader(dataset_training, batch_size=1, shuffle=False, num_workers=1)

# endregion CREATE_PATTERNS

# region CREATE_CONCEPTORS

# Create a set of conceptors
conceptors = ecnc.ConceptorSet(input_dim=reservoir_size)

# Create sixteen conceptors
for p in range(n_patterns):
    conceptors.add(0, ecnc.Conceptor(input_dim=reservoir_size, aperture=aperture, dtype=dtype))
# end for

# Create a conceptor network using
# an incrementing self-predicting ESN which
# will learn sixteen conceptors
conceptor_net = ecnc.IncConceptorNet(
    input_dim=1,
    hidden_dim=reservoir_size,
    output_dim=1,
    learning_algo='inv',
    w_generator=w_generator,
    win_generator=win_generator,
    wbias_generator=wbias_generator,
    input_scaling=1.0,
    ridge_param=ridge_param_wout,
    washout=washout_length,
    conceptor=conceptors,
    dtype=dtype
)

# endregion CREATE_CONCEPTORS

# region INCREMENTAL_LOADING

# Save pattern for plotting
P_collector = torch.empty(n_patterns, signal_plot_length, dtype=dtype)

# Save last state
last_states = torch.empty(n_patterns, reservoir_size, dtype=dtype)

# Save A after each loading
A_collector = list()

# Save quota after each loading
quota_collector = torch.zeros(n_patterns)

# Go through dataset
for p, data in enumerate(patterns_loader):
    # Inputs and labels
    inputs, outputs, labels = data

    # To Variable
    if dtype == torch.float64:
        inputs, outputs = Variable(inputs.double()), Variable(outputs.double())
    # end if

    # Set conceptor to use
    conceptors.set(p)

    # Feed SP-ESN
    X = conceptor_net(inputs, inputs)

    # Save pattern and states
    P_collector[p] = inputs[0, washout_length:washout_length+signal_plot_length, 0]
    last_states[p] = X[0, -1]
    quota_collector[p] = conceptors.quota()

    # Save A
    if p != 0:
        A_collector.append(conceptors.A())
    else:
        A_collector.append(ecnc.Conceptor.identity(reservoir_size))
    # end if
# end for

# endregion INCREMENTAL_LOADING

# region TEST

# In evaluation mode
conceptor_net.train(False)

# Generated patterns aligned
generated_samples_aligned = np.zeros((n_patterns, signal_plot_length))

# Save NRMSE of aligned patterns
NRMSEs_aligned = torch.zeros(n_patterns)

# Set conceptors in evaluation mode and generate a sample
for p in range(n_patterns):
    # Set it as current conceptor
    conceptors.set(p)

    # Set last state in training phase as initial
    # state here
    conceptor_net.cell.set_hidden(last_states[p])

    # Generate sample
    generated_sample = conceptor_net(torch.zeros(1, conceptor_test_length, 1, dtype=dtype), reset_state=False)

    # Find best phase shift
    generated_sample_aligned, _, NRMSE_aligned = echotorch.utils.pattern_interpolation(
        P_collector[p],
        generated_sample[0],
        interpolation_rate
    )

    # Save
    generated_samples_aligned[p] = generated_sample_aligned
    NRMSEs_aligned[p] = NRMSE_aligned
# end for

# Show NRMSE
print(u"Average NRMSE except last : {}".format(torch.mean(NRMSEs_aligned[:-1])))

# endregion TEST

# region PLOT

# Figure (square size)
plt.figure(figsize=(16, 16))

# Plot index
plot_index = 0

# For each pattern
for p in range(n_patterns):
    # Plot 1 : original pattern and recreated pattern
    plt.subplot(4, 4, plot_index + 1)
    plot_index += 1

    # C(all) after learning the pattern
    A = A_collector[p]
    _, Sx, _ = A.SVD

    # Plot singular values of C(all)
    plt.fill(np.linspace(0, signal_plot_length, n_plot_singular_values), 2.0 * Sx - 1.0, color='red', alpha=0.75)
    plt.fill_between(np.linspace(0, signal_plot_length, n_plot_singular_values), 2.0 * Sx - 1.0, -1, color='red', alpha=0.75)

    # Plot generated pattern and original
    plt.plot(generated_samples_aligned[p], color='lime', linewidth=10)
    plt.plot(P_collector[p], color='black', linewidth=1.5)

    # Square properties
    plot_width = signal_plot_length
    plot_bottom = -1
    plot_top = 1
    props = dict(boxstyle='square', facecolor='white', alpha=0.75)

    # Pattern number
    plt.text(
        plot_width - 0.7,
        plot_top - 0.1,
        "p = {}".format(p),
        fontsize=14,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
    )

    # Aligned NRMSE
    plt.text(
        plot_width - 0.7,
        plot_bottom + 0.1,
        round(NRMSEs_aligned[p].item(), 4),
        fontsize=14,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=props
    )

    # Show C(all) size
    plt.text(
        0.7, plot_bottom + 0.1,
        round(quota_collector[p].item(), 2),
        fontsize=14,
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=props
    )

    # Title
    if p == 0:
        plt.title(u'p and y')
    # end if

    # Title
    if p == 0:
        plt.title(u'p and y')
    # end if

    # X labels
    if p == 3:
        plt.xticks([0, signal_plot_length / 2.0, signal_plot_length])
    else:
        plt.xticks([0, signal_plot_length / 2.0, signal_plot_length])
    # end if

    # Y limits
    plt.ylim([-1, 1])
    plt.xlim([0, signal_plot_length])
    plt.yticks([-1, 0, 1])
# end for

# Show figure
plt.show()

# endregion PLOT
