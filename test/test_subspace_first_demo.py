# -*- coding: utf-8 -*-
#
# File : test/test_subspace_first_demo.py
# Description : Test reservoir loading and conceptor learning.
# Date : 3th of November, 2019
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


# Test case : Subspace first demo.
class Test_Subspace_First_Demo(EchoTorchTestCase):
    """
    Test subspace first demo
    """

    # region PUBLIC

    # Subspace first demo
    def subspace_first_demo(self, data_dir, expected_training_NRMSE, expected_average_NRMSEs, reservoir_size=100,
                            spectral_radius=1.5, input_scaling=1.5, bias_scaling=0.2, washout_length=500,
                            learn_length=1000, ridge_param_wstar=0.0001, ridge_param_wout=0.01, aperture=10,
                            connectivity=10.0, loading_method=ecnc.SPESNCell.W_LOADING, places=3, torch_seed=1,
                            np_seed=1, use_matlab_params=True, expected_RSim=None, expected_CSim=None,
                            dtype=torch.float64):
        """
        Subspace first demo
        :param data_dir: Directory in the test directory
        :param reservoir_size:
        :param spectral_radius:
        :param input_scaling:
        :param bias_scaling:
        :param connectivity:
        :param washout_length:
        :param learn_length:
        :param ridge_param_wstar:
        :param ridge_param_wout:
        :param aperture:
        :param loading_method:
        :param places:
        :param torch_seed:
        :param np_seed:
        :param dtype:
        """
        # Package
        subpackage_dir, this_filename = os.path.split(__file__)
        package_dir = os.path.join(subpackage_dir, "..")
        TEST_PATH = os.path.join(package_dir, "data", "tests", data_dir)

        # Debug ?
        debug_mode = Node.DEBUG_TEST_CASE

        # Precision decimal
        precision = 1.0 / places
        # precision_decimals = int(-np.log10(precision))

        # Init random number generators
        torch.random.manual_seed(torch_seed)
        np.random.seed(np_seed)

        # Reservoir parameters
        connectivity = connectivity / reservoir_size

        # Patterns parameters
        n_patterns = 4

        # Test parameters
        signal_plot_length = 20
        conceptor_test_length = 200
        interpolation_rate = 20

        # Load W from matlab file and random init ?
        if use_matlab_params:
            # Load internal weights W*
            wstar_generator = mg.matrix_factory.get_generator(
                "matlab",
                file_name=os.path.join(TEST_PATH, "WstarRaw.mat"),
                entity_name="WstarRaw",
                scale=spectral_radius
            )

            # Load input-internal weights
            win_generator = mg.matrix_factory.get_generator(
                "matlab",
                file_name=os.path.join(TEST_PATH, "WinRaw.mat"),
                entity_name="WinRaw",
                scale=input_scaling
            )

            # Load bias
            wbias_generator = mg.matrix_factory.get_generator(
                "matlab",
                file_name=os.path.join(TEST_PATH, "Wbias.mat"),
                entity_name="Wbias",
                shape=reservoir_size,
                scale=bias_scaling
            )
        else:
            # Generate internal weights
            wstar_generator = mg.matrix_factory.get_generator(
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

            # Load Wbias from matlab from or init randomly
            wbias_generator = mg.matrix_factory.get_generator(
                "normal",
                mean=0.0,
                std=1.0,
                connectivity=1.0,
                scale=bias_scaling
            )
        # end if

        # Four pattern (two sine, two periodic)
        pattern1_training = etds.SinusoidalTimeseries(
            sample_len=washout_length + learn_length,
            n_samples=1, a=1,
            period=8.8342522,
            dtype=dtype
        )

        pattern2_training = etds.SinusoidalTimeseries(
            sample_len=washout_length + learn_length,
            n_samples=1,
            a=1,
            period=9.8342522,
            dtype=dtype
        )

        pattern3_training = etds.PeriodicSignalDataset(
            sample_len=washout_length + learn_length,
            n_samples=1,
            period=[0.9000000000000002, -0.11507714997817164, 0.17591170369788622, -0.9, -0.021065045054201592],
            dtype=dtype
        )

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
        conceptors = ecnc.ConceptorSet(input_dim=reservoir_size)

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
            test_case=self,
            learning_algo='inv',
            w_generator=wstar_generator,
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

        # Use matlab data ?
        if use_matlab_params:
            # Load sample matrices
            for i in range(n_patterns):
                # Input patterns
                conceptor_net.cell.debug_point(
                    "u{}".format(i),
                    torch.reshape(torch.from_numpy(np.load(os.path.join(TEST_PATH, "u{}.npy".format(i)))), shape=(-1, 1)),
                    precision
                )

                # States
                conceptor_net.cell.debug_point(
                    "X{}".format(i),
                    torch.from_numpy(np.load(os.path.join(TEST_PATH, "X{}.npy".format(i)))),
                    precision
                )

                # Targets
                if loading_method == ecnc.SPESNCell.W_LOADING:
                    conceptor_net.cell.debug_point(
                        "Y{}".format(i),
                        torch.from_numpy(np.load(os.path.join(TEST_PATH, "Y{}.npy".format(i)))),
                        precision
                    )
                # end if

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

            # Load debug W, xTx, xTy
            conceptor_net.cell.debug_point(
                "Wstar",
                torch.from_numpy(np.load(os.path.join(TEST_PATH, "Wstar.npy"), allow_pickle=True)),
                precision
            )
            conceptor_net.cell.debug_point("Win", torch.from_numpy(np.load(os.path.join(TEST_PATH, "Win.npy"))), precision)
            conceptor_net.cell.debug_point("Wbias", torch.from_numpy(np.load(os.path.join(TEST_PATH, "Wbias.npy"))), precision)
            conceptor_net.cell.debug_point("xTx", torch.from_numpy(np.load(os.path.join(TEST_PATH, "xTx.npy"))), precision)
            conceptor_net.cell.debug_point("w_ridge_param", 0.0001, precision)
            conceptor_net.cell.debug_point(
                "ridge_xTx",
                torch.from_numpy(np.load(os.path.join(TEST_PATH, "ridge_xTx.npy"))),
                precision
            )
            conceptor_net.cell.debug_point(
                "inv_xTx",
                torch.from_numpy(np.load(os.path.join(TEST_PATH, "inv_xTx.npy"))),
                precision
            )
            conceptor_net.cell.debug_point("w", torch.from_numpy(np.load(os.path.join(TEST_PATH, "W.npy"))), precision)

            # Debug not related to inputs recreation
            if loading_method == ecnc.SPESNCell.W_LOADING:
                conceptor_net.cell.debug_point(
                    "xTy",
                    torch.from_numpy(np.load(os.path.join(TEST_PATH, "xTy.npy"))),
                    precision
                )
            # end if
        # end if

        # Xold and Y collectors
        Xold_collector = torch.empty(n_patterns * learn_length, reservoir_size, dtype=dtype)
        Y_collector = torch.empty(n_patterns * learn_length, reservoir_size, dtype=dtype)
        P_collector = torch.empty(n_patterns, signal_plot_length, dtype=dtype)

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
        # end for

        # Learn internal weights
        conceptor_net.finalize()

        # Predicted by W
        predY = torch.mm(conceptor_net.cell.w, Xold_collector.t()).t()

        # Compute NRMSE
        training_NRMSE = echotorch.utils.nrmse(predY, Y_collector)

        # Check training NRMSE
        self.assertLessEqual(training_NRMSE, expected_training_NRMSE)

        # No washout this time
        conceptor_net.washout = 0

        # Conceptors ON
        conceptor_net.conceptor_active(True)

        # NRMSE between original and aligned pattern
        NRMSEs_aligned = torch.zeros(n_patterns)

        # Train conceptors (Compute C from R)
        conceptors.finalize()

        # Set conceptors in evaluation mode and generate a sample
        for i in range(n_patterns):
            # Set it as current conceptor
            conceptors.set(i)

            # Randomly generated initial state (x0)
            conceptor_net.cell.set_hidden(0.5 * torch.randn(reservoir_size, dtype=dtype))

            # Generate sample
            generated_sample = conceptor_net(torch.zeros(1, conceptor_test_length, 1, dtype=dtype), reset_state=False)

            # Find best phase shift
            generated_sample_aligned, _, NRMSE_aligned = echotorch.utils.pattern_interpolation(
                P_collector[i],
                generated_sample[0],
                interpolation_rate
            )

            # Save NRMSE
            NRMSEs_aligned[i] = NRMSE_aligned
        # end for

        # Check NRMSE
        self.assertLessEqual(torch.mean(NRMSEs_aligned).item(), expected_average_NRMSEs)

        # Compute similarity matrices
        Rsim_test = conceptors.similarity_matrix(based_on='R')
        Csim_test = conceptors.similarity_matrix(based_on='C')

        # Load similarity matrices
        if use_matlab_params:
            Rsim = torch.from_numpy(np.load(os.path.join(TEST_PATH, "Rsim.npy")))
            Csim = torch.from_numpy(np.load(os.path.join(TEST_PATH, "Csim.npy")))
        else:
            Rsim = expected_RSim
            Csim = expected_CSim
        # end if

        # Test similarity matrices
        self.assertTensorAlmostEqual(Rsim_test, Rsim, precision)
        self.assertTensorAlmostEqual(Csim_test, Csim, precision)
    # subspace_first_demo

    # endregion PUBLIC

    # region TEST

    # Subspace first demo with matlab
    def test_subspace_first_demo_w_loading_matlab(self):
        """
        Subspace first demo
        """
        self.subspace_first_demo(
            data_dir="subspace_first_demo",
            use_matlab_params=True,
            loading_method=ecnc.SPESNCell.W_LOADING,
            expected_training_NRMSE=0.03,
            expected_average_NRMSEs=0.02
        )
    # end test_subspace_first_demo_matlab

    # Subspace first demo (input simulation) with matlab
    def test_subspace_first_demo_input_simulation_matlab(self):
        """
        Subspace first demo
        """
        self.subspace_first_demo(
            data_dir="subspace_first_demo",
            use_matlab_params=True,
            loading_method=ecnc.SPESNCell.INPUTS_SIMULATION,
            expected_training_NRMSE=0.8,
            expected_average_NRMSEs=0.02
        )
    # end test_subspace_first_demo_input_simulation_matlab

    # Subspace first demo (input recreation) with matlab
    def test_subspace_first_demo_input_recreation_matlab(self):
        """
        Subspace first demo
        """
        self.subspace_first_demo(
            data_dir="subspace_first_demo",
            use_matlab_params=True,
            loading_method=ecnc.SPESNCell.INPUTS_RECREATION,
            expected_training_NRMSE=0.8,
            expected_average_NRMSEs=0.02
        )
    # end test_subspace_first_demo_input_recreation_matlab

    # Subspace first demo with 100 neurons
    def test_subspace_first_demo_w_loading_100neurons(self):
        """
        Subspace first demo with 100 neurons
        """
        self.subspace_first_demo(
            data_dir="subspace_first_demo",
            use_matlab_params=False,
            loading_method=ecnc.SPESNCell.W_LOADING,
            expected_training_NRMSE=0.04,
            expected_average_NRMSEs=0.02,
            torch_seed=1,
            np_seed=1,
            expected_RSim=torch.tensor(
                [
                    [1.0000, 0.9955, 0.6086, 0.6500],
                    [0.9955, 1.0000, 0.5908, 0.6286],
                    [0.6086, 0.5908, 1.0000, 0.9736],
                    [0.6500, 0.6286, 0.9736, 1.0000]
                ]),
            expected_CSim=torch.tensor(
                [
                    [1.0000, 0.8578, 0.3633, 0.3793],
                    [0.8578, 1.0000, 0.3715, 0.3875],
                    [0.3633, 0.3715, 1.0000, 0.9667],
                    [0.3793, 0.3875, 0.9667, 1.0000]
                ]
            )
        )
    # end test_subspace_first_demo_w_loading_100neurons

    # Subspace first demo with 100 neurons
    def test_subspace_first_demo_w_loading_100neurons_32bits(self):
        """
        Subspace first demo with 100 neurons
        """
        self.subspace_first_demo(
            data_dir="subspace_first_demo",
            use_matlab_params=False,
            loading_method=ecnc.SPESNCell.W_LOADING,
            places=1,
            expected_training_NRMSE=0.1,
            expected_average_NRMSEs=0.1,
            torch_seed=1,
            np_seed=1,
            dtype=torch.float32,
            expected_RSim=torch.tensor(
                [
                    [1.0000, 0.9818, 0.4404, 0.5243],
                    [0.9818, 1.0000, 0.4115, 0.4862],
                    [0.4404, 0.4115, 1.0000, 0.9337],
                    [0.5243, 0.4862, 0.9337, 1.0000]
                ]),
            expected_CSim=torch.tensor(
                [
                    [1.0000, 0.6777, 0.2949, 0.2932],
                    [0.6777, 1.0000, 0.2448, 0.2380],
                    [0.2949, 0.2448, 1.0000, 0.9267],
                    [0.2932, 0.2380, 0.9267, 1.0000]
                ]
            )
        )
    # end test_subspace_first_demo_w_loading_100neurons_32bits

    # Subspace first demo with 100 neurons, no washout
    def test_subspace_first_demo_w_loading_100neurons_nowashout(self):
        """
        Subspace first demo with 100 neurons
        """
        self.subspace_first_demo(
            data_dir="subspace_first_demo",
            washout_length=0,
            use_matlab_params=False,
            loading_method=ecnc.SPESNCell.W_LOADING,
            expected_training_NRMSE=0.02,
            expected_average_NRMSEs=0.02,
            torch_seed=1,
            np_seed=1,
            expected_RSim=torch.tensor(
                [
                    [1.0000, 0.9955, 0.6096, 0.6510],
                    [0.9955, 1.0000, 0.5917, 0.6296],
                    [0.6096, 0.5917, 1.0000, 0.9736],
                    [0.6510, 0.6296, 0.9736, 1.0000]
                ]),
            expected_CSim=torch.tensor(
                [
                    [1.0000, 0.8743, 0.4223, 0.4346],
                    [0.8743, 1.0000, 0.4293, 0.4424],
                    [0.4223, 0.4293, 1.0000, 0.9589],
                    [0.4346, 0.4424, 0.9589, 1.0000]
                ]
            )
        )
    # end test_subspace_first_demo_w_loading_100neurons_nowashout

    # Subspace first demo (input simulation) with 100 neurons
    def test_subspace_first_demo_input_simulation_100neurons(self):
        """
        Subspace first demo with 100 neurons
        """
        self.subspace_first_demo(
            data_dir="subspace_first_demo",
            use_matlab_params=False,
            loading_method=ecnc.SPESNCell.INPUTS_SIMULATION,
            expected_training_NRMSE=0.8,
            expected_average_NRMSEs=0.02,
            torch_seed=1,
            np_seed=1,
            expected_RSim=torch.tensor(
                [
                    [1.0000, 0.9955, 0.6086, 0.6500],
                    [0.9955, 1.0000, 0.5908, 0.6286],
                    [0.6086, 0.5908, 1.0000, 0.9736],
                    [0.6500, 0.6286, 0.9736, 1.0000]
                ]),
            expected_CSim=torch.tensor(
                [
                    [1.0000, 0.8578, 0.3633, 0.3793],
                    [0.8578, 1.0000, 0.3715, 0.3875],
                    [0.3633, 0.3715, 1.0000, 0.9667],
                    [0.3793, 0.3875, 0.9667, 1.0000]
                ]
            )
        )
    # end test_subspace_first_demo_input_simulation_100neurons

    # Subspace first demo (input simulation) with 100 neurons (32 bits)
    def test_subspace_first_demo_input_simulation_100neurons_32bits(self):
        """
        Subspace first demo with 100 neurons
        """
        self.subspace_first_demo(
            data_dir="subspace_first_demo",
            use_matlab_params=False,
            loading_method=ecnc.SPESNCell.INPUTS_SIMULATION,
            expected_training_NRMSE=0.8,
            expected_average_NRMSEs=0.1,
            places=1,
            torch_seed=1,
            np_seed=1,
            dtype=torch.float32,
            expected_RSim=torch.tensor(
                [
                    [1.0000, 0.9818, 0.4404, 0.5243],
                    [0.9818, 1.0000, 0.4115, 0.4862],
                    [0.4404, 0.4115, 1.0000, 0.9337],
                    [0.5243, 0.4862, 0.9337, 1.0000]
                ]),
            expected_CSim=torch.tensor(
                [
                    [1.0000, 0.6777, 0.2949, 0.2932],
                    [0.6777, 1.0000, 0.2448, 0.2380],
                    [0.2949, 0.2448, 1.0000, 0.9267],
                    [0.2932, 0.2380, 0.9267, 1.0000]
                ]
            )
        )
    # end test_subspace_first_demo_input_simulation_100neurons_32bits

    # Subspace first demo (input recreation) with 100 neurons
    def test_subspace_first_demo_input_recreation_100neurons(self):
        """
        Subspace first demo with 100 neurons
        """
        self.subspace_first_demo(
            data_dir="subspace_first_demo",
            use_matlab_params=False,
            loading_method=ecnc.SPESNCell.INPUTS_RECREATION,
            expected_training_NRMSE=0.8,
            expected_average_NRMSEs=0.02,
            torch_seed=1,
            np_seed=1,
            expected_RSim=torch.tensor(
                [
                    [1.0000, 0.9955, 0.6086, 0.6500],
                    [0.9955, 1.0000, 0.5908, 0.6286],
                    [0.6086, 0.5908, 1.0000, 0.9736],
                    [0.6500, 0.6286, 0.9736, 1.0000]
                ]),
            expected_CSim=torch.tensor(
                [
                    [1.0000, 0.8578, 0.3633, 0.3793],
                    [0.8578, 1.0000, 0.3715, 0.3875],
                    [0.3633, 0.3715, 1.0000, 0.9667],
                    [0.3793, 0.3875, 0.9667, 1.0000]
                ]
            )
        )
    # end test_subspace_first_demo_input_recreation_100neurons

    # Subspace first demo (input recreation) with 100 neurons (32 bits)
    def test_subspace_first_demo_input_recreation_100neurons_32bits(self):
        """
        Subspace first demo with 100 neurons (32 bits)
        """
        self.subspace_first_demo(
            data_dir="subspace_first_demo",
            use_matlab_params=False,
            loading_method=ecnc.SPESNCell.INPUTS_RECREATION,
            expected_training_NRMSE=0.7,
            expected_average_NRMSEs=0.1,
            places=1,
            torch_seed=1,
            np_seed=1,
            dtype=torch.float32,
            expected_RSim=torch.tensor(
                [
                    [1.0000, 0.9818, 0.4404, 0.5243],
                    [0.9818, 1.0000, 0.4115, 0.4862],
                    [0.4404, 0.4115, 1.0000, 0.9337],
                    [0.5243, 0.4862, 0.9337, 1.0000]
                ]),
            expected_CSim=torch.tensor(
                [
                    [1.0000, 0.6777, 0.2949, 0.2932],
                    [0.6777, 1.0000, 0.2448, 0.2380],
                    [0.2949, 0.2448, 1.0000, 0.9267],
                    [0.2932, 0.2380, 0.9267, 1.0000]
                ]
            )
        )
    # end test_subspace_first_demo_input_recreation_100neurons_32bits

    # endregion TEST

# end Test_Subspace_First_Demo
