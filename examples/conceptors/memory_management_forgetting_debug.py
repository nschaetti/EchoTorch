# -*- coding: utf-8 -*-
#
# File : examples/conceptors/memory_management_debug.py
# Description : Conceptor memory management demo with debug mode
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
from echotorch.datasets import DatasetComposer
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
from echotorch.nn import Node

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
torch.random.manual_seed(5)
np.random.seed(5)

# region PARAMS

# 1. In this section, we set all the different parameters
# used in this experiment.

# Type params
dtype=torch.float64

# Reservoir parameters
reservoir_size = 100
spectral_radius = 1.5
bias_scaling = 0.25
connectivity = 10.0 / reservoir_size

# Inputs parameters
input_scaling = 1.5

# Washout and learning lengths
washout_length = 100
learn_length = 100
loading_method = ecnc.SPESNCell.INPUTS_SIMULATION
forgetting_version = ecnc.IncForgSPESNCell.FORGETTING_E

# Testing parameters
interpolation_rate = 20
conceptor_test_length = 200

# Regularization parameters
ridge_param_wout = 0.01

# Ploting parameters
signal_plot_length = 20
n_plot_singular_values = 100

# Aperture parameter
aperture = 1000

# Pattern parameters
n_patterns = 16

# endregion PARAMS

# Argument parsing
parser = argparse.ArgumentParser(
    prog="memory_management_incremental_forgetting",
    description="Memory management + incremental forgetting (with debug)"
)
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

# 2. We generate matrices Wstar, Win and Wbias,
# either from a random number generator or from
# matlab files.

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

# 3. We create the different patterns to be loaded
# into the reservoir and learned by the Conceptors.
# There is 13 patterns, 3 are repeated (6, 7, 8)
# to show that it does not increase memory size.

# Pattern 1 (sine p=10)
pattern1_training = etds.SinusoidalTimeseries(
    sample_len=washout_length + learn_length,
    n_samples=1,
    a=1,
    period=10,
    dtype=dtype
)

# Pattern 2 (sine p=15)
pattern2_training = etds.SinusoidalTimeseries(
    sample_len=washout_length + learn_length,
    n_samples=1,
    a=1,
    period=15,
    dtype=dtype
)

# Pattern 3 (periodic 4)
pattern3_training = etds.PeriodicSignalDataset(
    sample_len=washout_length + learn_length,
    n_samples=1,
    period=[-0.4564, 0.6712, -2.3953, -2.1594],
    dtype=dtype
)

# Pattern 4 (periodic 6)
pattern4_training = etds.PeriodicSignalDataset(
    sample_len=washout_length + learn_length,
    n_samples=1,
    period=[0.5329, 0.9621, 0.1845, 0.5099, 0.3438, 0.7697],
    dtype=dtype
)

# Pattern 5 (periodic 7)
pattern5_training = etds.PeriodicSignalDataset(
    sample_len=washout_length + learn_length,
    n_samples=1,
    period=[0.8029, 0.4246, 0.2041, 0.0671, 0.1986, 0.2724, 0.5988],
    dtype=dtype
)

# Pattern 6 (sine p=12)
pattern6_training = etds.SinusoidalTimeseries(
    sample_len=washout_length + learn_length,
    n_samples=1,
    a=1,
    period=12,
    dtype=dtype
)

# Pattern 7 (sine p=5)
pattern7_training = etds.SinusoidalTimeseries(
    sample_len=washout_length + learn_length,
    n_samples=1,
    a=1,
    period=5,
    dtype=dtype
)

# Pattern 8 (sine p=6)
pattern8_training = etds.SinusoidalTimeseries(
    sample_len=washout_length + learn_length,
    n_samples=1,
    a=1,
    period=6,
    dtype=dtype
)

# Pattern 9 (periodic 8)
pattern9_training = etds.PeriodicSignalDataset(
    sample_len=washout_length + learn_length,
    n_samples=1,
    period=[0.8731, 0.1282, 0.9582, 0.6832, 0.7420, 0.9829, 0.4161, 0.5316],
    dtype=dtype
)

# Pattern 10 (periodic 7)
pattern10_training = etds.PeriodicSignalDataset(
    sample_len=washout_length + learn_length,
    n_samples=1,
    period=[0.6792, 0.5129, 0.2991, 0.1054, 0.2849, 0.7689, 0.6408],
    dtype=dtype
)

# Pattern 11 (periodic 3)
pattern11_training = etds.PeriodicSignalDataset(
    sample_len=washout_length + learn_length,
    n_samples=1,
    period=[1.4101, -0.0992, -0.0902],
    dtype=dtype
)

# Pattern 12 (sine p=6)
pattern12_training = etds.SinusoidalTimeseries(
    sample_len=washout_length + learn_length,
    n_samples=1,
    a=1,
    period=11,
    dtype=dtype
)

# Pattern 13 (periodic 5)
pattern13_training = etds.PeriodicSignalDataset(
    sample_len=washout_length + learn_length,
    n_samples=1,
    period=[0.9, -0.021439412841318672, 0.0379515995051003, -0.9, 0.06663989939293802],
    dtype=dtype
)

# Composer
dataset_training = DatasetComposer([
    pattern1_training, pattern2_training, pattern3_training, pattern4_training, pattern5_training, pattern1_training,
    pattern2_training, pattern3_training, pattern6_training, pattern7_training, pattern8_training, pattern9_training,
    pattern10_training, pattern11_training, pattern12_training, pattern13_training
])

# Data loader
patterns_loader = DataLoader(dataset_training, batch_size=1, shuffle=False, num_workers=1)

# endregion CREATE_PATTERNS

# region CREATE_CONCEPTORS

# 4. We create a conceptor set, 16 conceptors,
# and an incremental conceptor net (IncConceptorNet)

# Create a set of conceptors
conceptors = ecnc.ConceptorSet(input_dim=reservoir_size, debug=debug_mode)

# Create sixteen conceptors
for p in range(n_patterns):
    conceptors.add(p, ecnc.Conceptor(input_dim=reservoir_size, aperture=aperture, debug=debug_mode, dtype=dtype))
# end for

# Create a conceptor network using
# an incrementing self-predicting ESN which
# will learn sixteen patterns
conceptor_net = ecnc.IncConceptorNet(
    input_dim=1,
    hidden_dim=reservoir_size,
    output_dim=1,
    conceptor=conceptors,
    w_generator=w_generator,
    win_generator=win_generator,
    wbias_generator=wbias_generator,
    ridge_param_wout=ridge_param_wout,
    aperture=aperture,
    washout=washout_length,
    fill_left=True,
    loading_method=loading_method,
    incremental_forgetting=True,
    forgetting_lambda=0.0,
    forgetting_version=forgetting_version,
    debug=debug_mode,
    dtype=dtype
)

# endregion CREATE_CONCEPTORS

# region DEBUG

# 6. For debugging, we add some debug point
# which will compute the differences between
# what we want and what we have. You will then be
# able to check if there is a problem.

# If in debug mode
if debug_mode > Node.NO_DEBUG:
    # Load sample matrices
    for i in range(n_patterns):
        # SPESN : Input patterns
        conceptor_net.cell.debug_point(
            "u{}".format(i),
            torch.reshape(torch.from_numpy(np.load("data/debug/memory_management/u{}.npy".format(i))), shape=(-1, 1)),
            precision
        )

        # SPESN : States X
        conceptor_net.cell.debug_point(
            "X{}".format(i),
            torch.from_numpy(np.load("data/debug/memory_management/X{}.npy".format(i)).T),
            precision
        )

        # SPESN : States old
        conceptor_net.cell.debug_point(
            "Xold{}".format(i),
            torch.from_numpy(np.load("data/debug/memory_management/XOld{}.npy".format(i)).T),
            precision
        )

        # SPESN : Td
        if loading_method != ecnc.SPESNCell.INPUTS_RECREATION:
            conceptor_net.cell.debug_point(
                "Td{}".format(i),
                torch.from_numpy(np.load("data/debug/memory_management/Td{}.npy".format(i)).T),
                precision
            )
        # end if

        # SPESN : F
        conceptor_net.cell.debug_point(
            "F{}".format(i),
            torch.from_numpy(np.load("data/debug/memory_management/F{}.npy".format(i))),
            precision
        )

        # SPESN : Sold
        conceptor_net.cell.debug_point(
            "Sold{}".format(i),
            torch.from_numpy(np.load("data/debug/memory_management/Sold{}.npy".format(i)).T),
            precision
        )

        # SPESN : sTd
        if loading_method != ecnc.SPESNCell.INPUTS_RECREATION:
            conceptor_net.cell.debug_point(
                "sTd{}".format(i),
                torch.from_numpy(np.load("data/debug/memory_management/sTd{}.npy".format(i))),
                precision
            )
        # end if

        # SPESN : sTs
        conceptor_net.cell.debug_point(
            "sTs{}".format(i),
            torch.from_numpy(np.load("data/debug/memory_management/sTs{}.npy".format(i))),
            precision
        )

        # SPESN : ridge sTs
        conceptor_net.cell.debug_point(
            "ridge_sTs{}".format(i),
            torch.from_numpy(np.load("data/debug/memory_management/ridge_sTs{}.npy".format(i))),
            precision
        )

        # SPESN : Dinc
        conceptor_net.cell.debug_point(
            "Dinc{}".format(i),
            torch.from_numpy(np.load("data/debug/memory_management/Dinc{}.npy".format(i))),
            precision
        )

        # SPESN : D
        conceptor_net.cell.debug_point(
            "D{}".format(i),
            torch.from_numpy(np.load("data/debug/memory_management/D{}.npy".format(i))),
            precision
        )

        # Conceptor : C matrix
        conceptors[i].debug_point(
            "C",
            torch.from_numpy(np.load("data/debug/memory_management/C{}.npy".format(i))),
            precision
        )

        # IncRRCell : Wout Y
        conceptor_net.output.debug_point(
            "Y{}".format(i),
            torch.from_numpy(np.load("data/debug/memory_management/Y{}.npy".format(i)).T),
            precision
        )

        # IncRRCell : Wout y
        conceptor_net.output.debug_point(
            "y{}".format(i),
            torch.reshape(torch.from_numpy(np.load("data/debug/memory_management/u{}.npy".format(i))), shape=(-1, 1)),
            precision
        )

        # IncRRCell : Wout F
        conceptor_net.output.debug_point(
            "F{}".format(i),
            torch.from_numpy(np.load("data/debug/memory_management/Wout_F{}.npy".format(i))),
            precision
        )

        # IncRRCell : Wout S
        conceptor_net.output.debug_point(
            "S{}".format(i),
            torch.from_numpy(np.load("data/debug/memory_management/S{}.npy".format(i)).T),
            precision
        )

        # IncRRCell : Wout sTs
        conceptor_net.output.debug_point(
            "sTs{}".format(i),
            torch.from_numpy(np.load("data/debug/memory_management/Wout_sTs{}.npy".format(i))),
            precision
        )

        # IncRRCell : Wout sTy
        conceptor_net.output.debug_point(
            "sTy{}".format(i),
            torch.from_numpy(np.load("data/debug/memory_management/sTy{}.npy".format(i))),
            precision
        )

        # IncRRCell : Wout ridge sTs
        conceptor_net.output.debug_point(
            "ridge_sTs{}".format(i),
            torch.from_numpy(np.load("data/debug/memory_management/Wout_ridge_sTs{}.npy".format(i))),
            precision
        )

        # IncRRCell : Wout inverse of ridge sTs
        conceptor_net.output.debug_point(
            "inv_sTs{}".format(i),
            torch.from_numpy(np.load("data/debug/memory_management/Wout_inv_ridge_sTs{}.npy".format(i))),
            precision
        )

        # IncRRCell : Wout
        conceptor_net.output.debug_point(
            "w_out{}".format(i),
            torch.reshape(torch.from_numpy(np.load("data/debug/memory_management/Wout{}.npy".format(i))),
                          shape=(1, -1)),
            precision
        )
    # end for

    # Load debug W, xTx, xTy
    conceptor_net.cell.debug_point("Wstar", torch.from_numpy(np.load("data/debug/memory_management/Wstar.npy", allow_pickle=True)), precision)
    conceptor_net.cell.debug_point("Win", torch.from_numpy(np.load("data/debug/memory_management/Win.npy")), precision)
    conceptor_net.cell.debug_point("Wbias", torch.from_numpy(np.load("data/debug/memory_management/Wbias.npy")), precision)
# end if

# endregion debug

# region INCREMENTAL_LOADING

# 7. We incrementally load the patterns into the reservoir
# and we save the results for plotting and testing.

# Save pattern for plotting, last state, A and quota after each loading
P_collector = torch.empty(n_patterns, signal_plot_length, dtype=dtype)
last_states = torch.empty(n_patterns, reservoir_size, dtype=dtype)
A_collector = list()
quota_collector = torch.zeros(n_patterns)
DINC_collector = torch.empty(n_patterns, reservoir_size, reservoir_size, dtype=dtype)
DUP_collector = torch.empty(n_patterns, reservoir_size, reservoir_size, dtype=dtype)
RINC_collector = torch.empty(n_patterns, reservoir_size, 1, dtype=dtype)
RUP_collector = torch.empty(n_patterns, reservoir_size, 1, dtype=dtype)

# Conceptors activated in the loop
conceptor_net.conceptor_active(True)

# For each sample in the dataset
for p, data in enumerate(patterns_loader):
    # Inputs and labels
    inputs, outputs, labels = data

    # To Variable
    if dtype == torch.float64:
        inputs, outputs = Variable(inputs.double()), Variable(outputs.double())
    # end if

    # Set the conceptor activated in
    # the loop.
    conceptors.set(p)

    # Feed SPESN with inputs,
    # output learning to recreate the inputs
    # so the inputs are the targets.
    X = conceptor_net(inputs, inputs)

    # Finalize Conceptor by learning
    # the Conceptor matrix from the
    # neurons activation received in
    # the preceding line.
    conceptors[p].finalize()

    # We change the aperture of the Conceptor,
    # the Conceptor matrix C is modified.
    conceptors[p].aperture = aperture

    # We save the patterns to be plotted afterwards,
    # we save the last state to start the generation.
    # we also save the quota of the space used by the
    # patterns currently loaded in the reservoir.
    P_collector[p] = inputs[0, washout_length:washout_length+signal_plot_length, 0]
    last_states[p] = X[0, -1]
    quota_collector[p] = conceptors.quota()

    # Save A = C1 OR ... OR Cn
    # of the currently loaded patterns
    # (with the aperture set in the
    # parameters).
    A_collector.append(conceptors.A())

    # Save D or R
    """if loading_method == ecnc.IncSPESNCell.INPUTS_SIMULATION:
        DINC_collector[p] = conceptor_net.cell.Dinc
        DUP_collector[p] = conceptor_net.cell.Dup
        print("Dinc:")
        print(conceptor_net.cell.Dinc)
        print("Dup:")
        print(conceptor_net.cell.Dup)
        print("D:")
        print(conceptor_net.cell.D)
    else:
        RINC_collector[p] = conceptor_net.cell.Rinc
        RUP_collector[p] = conceptor_net.cell.Rup
    # end if"""
# end for

# endregion INCREMENTAL_LOADING

# region TEST

# 8. We test the system by generating signals,
# we align these with original patterns and
# we measure its performances with NRMSE.

# We are going to to some pattern
# generation, so we stop the learning
# and switch to the evaluation mode.
conceptor_net.train(False)

# We save the pattern generated that we will
# aligned to the true pattern by finding
# the best phase shift.
generated_samples_aligned = np.zeros((n_patterns, signal_plot_length))

# We also save the NRMSE of the aligned
# patterns to see how this system can remember
# the loaded patterns.
NRMSEs_aligned = torch.zeros(n_patterns)

# For each pattern we generate a sample by filtering the neurons
# activation with the selected Conceptor, we then align the
# generated sample to the real pattern by testing different
# phase shift and we save the result.
for p in range(n_patterns):
    # Set the current conceptor
    # corresponding to the pth pattern.
    conceptors.set(p)

    # Set last state in training phase as initial
    # state here.
    conceptor_net.cell.set_hidden(last_states[p])

    # Generate sample, we give a zero input of size
    # batch size x time length x number of inputs.
    # We don't reset the state as we set the initial state
    # just before.
    generated_sample = conceptor_net(torch.zeros(1, conceptor_test_length, 1, dtype=dtype), reset_state=False)

    # We find the best phase shift by interpolating the original
    # and the generated signal quadratically and trying different
    # shifts. We take the best under the NRMSE evaluation measure.
    generated_sample_aligned, _, NRMSE_aligned = echotorch.utils.pattern_interpolation(
        P_collector[p],
        generated_sample[0],
        interpolation_rate
    )

    # Save aligned sample and the corresponding NRMSE
    generated_samples_aligned[p] = generated_sample_aligned
    NRMSEs_aligned[p] = NRMSE_aligned
# end for

# Show the average NRMSE
print(u"Average NRMSE : {}".format(torch.mean(NRMSEs_aligned)))
print(u"Average NRMSE except last : {}".format(torch.mean(NRMSEs_aligned[:-1])))

# endregion TEST

# region PLOTTING

# 9. We plot the original patterns and the generated signal,
# and the singular values of A (OR of all conceptor after loading
# the pth pattern) to show the share of the reservoir used.
# We show the NRMSE (lower right) and the quota (lower left).

# Figure (square size)
plt.figure(figsize=(20, 16))

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

# endregion PLOTTING
