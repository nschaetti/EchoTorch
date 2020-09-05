# -*- coding: utf-8 -*-
#
# File : test/test_morphing_square.py
# Description : Test pattern morphing
# Date : 18th of December, 2019
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
import os
import echotorch.utils
from . import EchoTorchTestCase
import numpy as np
import torch
import echotorch.nn.conceptors as ecnc
import echotorch.utils.matrix_generation as mg
import echotorch.utils
import echotorch.datasets as etds
from echotorch.datasets import DatasetComposer
from echotorch.nn.Node import Node
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable


# Test case : morphing patterns
class Test_Morphing_Square(EchoTorchTestCase):
    """
    Test pattern morphing
    """
    # region BODY

    # region PUBLIC

    # Morphing square
    def morphing_square(self, data_dir, expected_training_NRMSEs, expected_upper_left, expected_upper_right,
                        expected_bottom_left, expected_bottom_right, reservoir_size=100, spectral_radius=1.5,
                        input_scaling=1.5, bias_scaling=0.2, connectivity=10.0, washout_length=500, learn_length=1000,
                        ridge_param_wout=0.01, ridge_param_wstar = 0.0001, aperture=10, precision=0.001,
                        torch_seed=1, np_seed=1, morphing_length = 30, morphing_washout = 190,
                        signal_plot_length=20, use_matlab_params=True):
        """
        Morphing square
        """
        # Package
        subpackage_dir, this_filename = os.path.split(__file__)
        package_dir = os.path.join(subpackage_dir, "..")
        TEST_PATH = os.path.join(package_dir, "data", "tests", data_dir)

        # Debug
        debug_mode = Node.DEBUG_TEST_CASE

        # Random numb. init
        torch.random.manual_seed(torch_seed)
        np.random.seed(np_seed)

        # Precision decimal
        precision_decimals = int(-np.log10(precision))

        # Type params
        dtype = torch.float64

        # ESN params
        connectivity = connectivity / reservoir_size

        # Plots
        n_plots = 9

        # Morphing
        min_mu = -0.5
        max_mu = 1.5

        # Number of pattern
        n_patterns = 4

        # Load W from matlab file and random init ?
        if use_matlab_params:
            # Load internal weights
            w_generator = mg.matrix_factory.get_generator(
                "matlab",
                file_name=os.path.join(TEST_PATH, "WstarRaw.mat"),
                entity_name="WstarRaw",
                scale=spectral_radius
            )

            # Load internal weights
            win_generator = mg.matrix_factory.get_generator(
                "matlab",
                file_name=os.path.join(TEST_PATH, "WinRaw.mat"),
                entity_name="WinRaw",
                scale=input_scaling
            )

            # Load Wbias from matlab from or init randomly
            wbias_generator = mg.matrix_factory.get_generator(
                "matlab",
                file_name=os.path.join(TEST_PATH, "WbiasRaw.mat"),
                entity_name="WbiasRaw",
                shape=reservoir_size,
                scale=bias_scaling
            )

            # Load x0 from matlab from or init randomly
            x0_generator = mg.matrix_factory.get_generator(
                "matlab",
                file_name=os.path.join(TEST_PATH, "x0_fig2.mat"),
                entity_name="x0",
                shape=reservoir_size
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

            # Generate Win
            win_generator = mg.matrix_factory.get_generator(
                "normal",
                mean=0.0,
                std=1.0,
                connectivity=1.0,
                scale=input_scaling
            )

            # Wbias
            wbias_generator = mg.matrix_factory.get_generator(
                "normal",
                mean=0.0,
                std=1.0,
                connectivity=1.0,
                scale=bias_scaling
            )

            # Starting state
            x0_generator = mg.matrix_factory.get_generator(
                "normal",
                mean=0.0,
                std=1.0,
                connectivity=1.0
            )
        # end if

        # Pattern 1 (sine)
        pattern1_training = etds.SinusoidalTimeseries(
            sample_len=washout_length + learn_length,
            n_samples=1,
            a=1,
            period=8.8342522,
            dtype=dtype
        )

        # Pattern 2 (Sine)
        pattern2_training = etds.SinusoidalTimeseries(
            sample_len=washout_length + learn_length,
            n_samples=1,
            a=1,
            period=9.8342522,
            dtype=dtype
        )

        # Pattern 3 (Periodic)
        pattern3_training = etds.PeriodicSignalDataset(
            sample_len=washout_length + learn_length,
            n_samples=1,
            period=[0.9000000000000002, -0.11507714997817164, 0.17591170369788622, -0.9, -0.021065045054201592],
            dtype=dtype
        )

        # Pattern 4
        pattern4_training = etds.PeriodicSignalDataset(
            sample_len=washout_length + learn_length,
            n_samples=1,
            period=[0.9, -0.021439412841318672, 0.0379515995051003, -0.9, 0.06663989939293802],
            dtype=dtype
        )

        # Composer
        dataset_training = DatasetComposer([pattern1_training, pattern2_training, pattern3_training, pattern4_training])

        # Data loader
        patterns_loader = DataLoader(dataset_training, batch_size=1, shuffle=False, num_workers=1)

        # Create a set of conceptors
        conceptors = ecnc.ConceptorSet(input_dim=reservoir_size, debug=debug_mode, test_case=self, dtype=dtype)

        # Create four conceptors, one for each pattern
        conceptors.add(0, ecnc.Conceptor(input_dim=reservoir_size, aperture=aperture, debug=debug_mode, test_case=self, dtype=dtype))
        conceptors.add(1, ecnc.Conceptor(input_dim=reservoir_size, aperture=aperture, debug=debug_mode, test_case=self, dtype=dtype))
        conceptors.add(2, ecnc.Conceptor(input_dim=reservoir_size, aperture=aperture, debug=debug_mode, test_case=self, dtype=dtype))
        conceptors.add(3, ecnc.Conceptor(input_dim=reservoir_size, aperture=aperture, debug=debug_mode, test_case=self, dtype=dtype))

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
            debug=debug_mode,
            test_case=self,
            dtype=dtype
        )

        # If in debug mode
        if use_matlab_params:
            # Load sample matrices
            for i in range(n_patterns):
                # Input patterns
                conceptor_net.cell.debug_point(
                    "u{}".format(i),
                    torch.reshape(torch.from_numpy(np.load(os.path.join(TEST_PATH, "u{}.npy".format(i)))),
                                  shape=(-1, 1)),
                    precision
                )

                # States
                conceptor_net.cell.debug_point(
                    "X{}".format(i),
                    torch.from_numpy(np.load(os.path.join(TEST_PATH, "X{}.npy".format(i)))),
                    precision
                )

                # Targets
                conceptor_net.cell.debug_point(
                    "Y{}".format(i),
                    torch.from_numpy(np.load(os.path.join(TEST_PATH, "Y{}.npy".format(i)))),
                    precision
                )

                # Xold
                conceptor_net.cell.debug_point(
                    "Xold{}".format(i),
                    torch.from_numpy(np.load(os.path.join(TEST_PATH, "Xold{}.npy".format(i)))),
                    precision
                )

                # Conceptor
                conceptors[i].debug_point(
                    "C",
                    torch.from_numpy(np.load(os.path.join(TEST_PATH, "C{}.npy".format(i)))),
                    precision
                )
            # end for

            # Load debug Wstar
            conceptor_net.cell.debug_point(
                "Wstar",
                torch.from_numpy(np.load(os.path.join(TEST_PATH, "Wstar.npy"), allow_pickle=True)),
                precision
            )

            # Load debug Win
            conceptor_net.cell.debug_point(
                "Win",
                torch.from_numpy(np.load(os.path.join(TEST_PATH, "Win.npy"))),
                precision
            )

            # Load debug Wbias
            conceptor_net.cell.debug_point(
                "Wbias",
                torch.from_numpy(np.load(os.path.join(TEST_PATH, "Wbias.npy"))),
                precision
            )

            # xTx
            conceptor_net.cell.debug_point(
                "xTx",
                torch.from_numpy(np.load(os.path.join(TEST_PATH, "xTx.npy"))),
                precision
            )

            # xTy
            conceptor_net.cell.debug_point(
                "xTy",
                torch.from_numpy(np.load(os.path.join(TEST_PATH, "xTy.npy"))),
                precision
            )

            # W ridge param
            conceptor_net.cell.debug_point("w_ridge_param", 0.0001, precision)

            # Ridge xTx
            conceptor_net.cell.debug_point(
                "ridge_xTx",
                torch.from_numpy(np.load(os.path.join(TEST_PATH, "ridge_xTx.npy"))),
                precision
            )

            # inv xTx
            conceptor_net.cell.debug_point(
                "inv_xTx",
                torch.from_numpy(np.load(os.path.join(TEST_PATH, "inv_xTx.npy"))),
                precision
            )

            # W
            conceptor_net.cell.debug_point(
                "w",
                torch.from_numpy(np.load(os.path.join(TEST_PATH, "W.npy"))),
                precision
            )
        # end if

        # Xold and Y collectors
        Xold_collector = torch.empty(n_patterns * learn_length, reservoir_size, dtype=dtype)
        Y_collector = torch.empty(n_patterns * learn_length, reservoir_size, dtype=dtype)
        P_collector = torch.empty(n_patterns, signal_plot_length, dtype=dtype)
        last_X = torch.empty(n_patterns, reservoir_size, dtype=dtype)

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

            # Feed SP-ESN
            X = conceptor_net(inputs, inputs)

            # Get targets
            Y = conceptor_net.cell.targets(X[0])

            # Get features
            Xold = conceptor_net.cell.features(X[0])

            # Save
            Xold_collector[i * learn_length:i * learn_length + learn_length] = Xold
            Y_collector[i * learn_length:i * learn_length + learn_length] = Y
            P_collector[i] = inputs[0, washout_length:washout_length + signal_plot_length, 0]
            last_X[i] = X[0, -1]
        # end for

        # Learn internal weights
        conceptor_net.finalize()

        # Predicted by W
        predY = torch.mm(conceptor_net.cell.w, Xold_collector.t()).t()

        # Compute NRMSE
        training_NRMSE = echotorch.utils.nrmse(predY, Y_collector)

        # No washout this time
        conceptor_net.washout = 0

        # Conceptors ON
        conceptor_net.conceptor_active(True)

        # Train conceptors (Compute C from R)
        conceptors.finalize()

        # Corresponding mixture vectors
        mixture_vectors = torch.empty((n_plots, n_plots, 1, n_patterns))

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

        # No washout this time
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

                # Test output
                if i == 0 and j == 0:
                    self.assertTensorAlmostEqual(generated_sample[0], expected_upper_left, precision_decimals)
                elif i == 8 and j == 0:
                    self.assertTensorAlmostEqual(generated_sample[0], expected_bottom_left, precision_decimals)
                elif i == 0 and j == 8:
                    self.assertTensorAlmostEqual(generated_sample[0], expected_upper_right, precision_decimals)
                elif i == 8 and j == 8:
                    self.assertTensorAlmostEqual(generated_sample[0], expected_bottom_right, precision_decimals)
                # end if

                # Save outputs
                plots[i, j] = generated_sample[0, :, 0]
            # end for
        # end for

        # Test training NRMSE
        self.assertAlmostEqual(training_NRMSE, expected_training_NRMSEs, precision_decimals)
    # end memory_management

    # endregion PUBLIC

    # region TEST

    # Morphing square with matlab info
    def test_morphing_square_matlab(self):
        """
        Morphing square with matlab info
        """
        # Test with matlab params
        self.morphing_square(
            data_dir="morphing_square",
            use_matlab_params=True,
            expected_training_NRMSEs=0.02937710594939207,
            expected_upper_left=torch.tensor(
                [
                    [ 0.1314], [ 0.8812], [-1.1254], [ 0.9506], [-0.7192], [ 0.3554], [ 0.6780], [-1.1495],
                    [ 0.9800], [-0.7492], [ 0.5540], [ 0.3879], [-1.1164], [ 1.0282], [-0.8646], [ 0.7457],
                    [-0.0096], [-0.9301], [ 1.0759], [-0.9703], [ 0.8242], [-0.2804], [-0.6422], [ 1.1026],
                    [-1.0208], [ 0.8678], [-0.5141], [-0.2674], [ 1.0576], [-1.0735]
                ]
            ),
            expected_upper_right=torch.tensor(
                [
                    [-0.1398], [-0.1457], [-1.1539], [ 0.5392], [ 1.0713], [-0.4417], [ 0.0478], [-1.1504],
                    [ 0.6642], [ 0.9761], [-0.5085], [ 0.0458], [-1.1381], [ 0.7277], [ 0.9024], [-0.5169],
                    [-0.0159], [-1.1348], [ 0.7749], [ 0.8355], [-0.5018], [-0.1038], [-1.1334], [ 0.8139],
                    [ 0.7744], [-0.4783], [-0.1997], [-1.1259], [ 0.8468], [ 0.7240]
                ]
            ),
            expected_bottom_left=torch.tensor(
                [
                    [ 1.0400], [ 0.6237], [-0.0256], [-0.7218], [-1.3186], [-1.0866], [-0.6655], [-0.0027],
                    [ 0.6961], [ 1.2492], [ 1.1051], [ 0.7134], [ 0.0966], [-0.5978], [-1.2525], [-1.1947],
                    [-0.7476], [-0.1668], [ 0.5260], [ 1.1650], [ 1.2246], [ 0.8340], [ 0.2974], [-0.3812],
                    [-1.0724], [-1.3138], [-0.8643], [-0.3684], [ 0.3355], [ 0.9845]
                ]
            ),
            expected_bottom_right=torch.tensor(
                [
                    [-0.7198], [-1.2835], [-1.2005], [-0.9035], [-0.4424], [ 0.0463], [ 0.7154], [ 1.2580],
                    [ 1.2239], [ 0.9519], [ 0.4831], [-0.0465], [-0.7198], [-1.2835], [-1.2005], [-0.9035],
                    [-0.4424], [ 0.0463], [ 0.7154], [ 1.2580], [ 1.2239], [ 0.9519], [ 0.4831], [-0.0465],
                    [-0.7198], [-1.2835], [-1.2005], [-0.9035], [-0.4424], [ 0.0463]
                ]
            )
        )
    # end test_morphing_square_matlab

    # endregion TEST

    # endregion BODY
# end Test_Morphing_Square
