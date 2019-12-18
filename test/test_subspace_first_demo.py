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
from .EchoTorchTestCase import EchoTorchTestCase
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
    def subspace_first_demo(self, data_dir, reservoir_size=100, spectral_radius=1.5, input_scaling=1.5, bias_scaling=0.2,
                            connectivity=10.0, washout_length=500, learn_length=1000, ridge_param_wstar=0.0001,
                            ridge_param_wout=0.01, aperture=10, precision=0.001, torch_seed=1, np_seed=1):
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
        :param precision:
        :param torch_seed:
        :param np_seed:
        :return:
        """
        # Package
        subpackage_dir, this_filename = os.path.split(__file__)
        package_dir = os.path.join(subpackage_dir, "..")
        TEST_PATH = os.path.join(package_dir, "data", "tests", data_dir)

        # Debug ?
        debug_mode = Node.DEBUG_TEST_CASE

        # Precision decimal
        precision_decimals = -np.log10(precision)

        # Init random number generators
        torch.random.manual_seed(torch_seed)
        np.random.seed(np_seed)

        # Parameters
        signal_plot_length = 20
        conceptor_test_length = 200
        interpolation_rate = 20
        dtype = torch.float64

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

        # Create a self-predicting ESN
        # which will be loaded with the
        # four patterns.
        spesn = ecnc.SPESN(
            input_dim=1,
            hidden_dim=reservoir_size,
            output_dim=1,
            learning_algo='inv',
            w_generator=wstar_generator,
            win_generator=win_generator,
            wbias_generator=wbias_generator,
            input_scaling=1.0,
            ridge_param=ridge_param_wout,
            w_ridge_param=ridge_param_wstar,
            washout=washout_length,
            debug=debug_mode,
            test_case=self,
            dtype=dtype
        )

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
            esn_cell=spesn.cell,
            conceptor=conceptors,
            test_case=self,
            dtype=dtype
        )

        # Load sample matrices
        for i in range(4):
            # Input patterns
            spesn.cell.debug_point(
                "u{}".format(i),
                torch.reshape(torch.from_numpy(np.load("data/tests/subspace_first_demo/u{}.npy".format(i))),
                              shape=(-1, 1)),
                precision
            )

            # States
            spesn.cell.debug_point(
                "X{}".format(i),
                torch.from_numpy(np.load("data/tests/subspace_first_demo/X{}.npy".format(i))),
                precision
            )

            # Targets
            spesn.cell.debug_point(
                "Y{}".format(i),
                torch.from_numpy(np.load("data/tests/subspace_first_demo/Y{}.npy".format(i))),
                precision
            )

            # Xold
            spesn.cell.debug_point(
                "Xold{}".format(i),
                torch.from_numpy(np.load("data/tests/subspace_first_demo/Xold{}.npy".format(i))),
                precision
            )

            # Conceptor
            conceptors[i].debug_point(
                "C",
                torch.from_numpy(np.load("data/tests/subspace_first_demo/C{}.npy".format(i))),
                precision
            )
        # end for

        # Load debug W, xTx, xTy
        spesn.cell.debug_point(
            "Wstar",
            torch.from_numpy(np.load("data/tests/subspace_first_demo/Wstar.npy", allow_pickle=True)),
            precision
        )
        spesn.cell.debug_point("Win", torch.from_numpy(np.load("data/tests/subspace_first_demo/Win.npy")), precision)
        spesn.cell.debug_point("Wbias", torch.from_numpy(np.load("data/tests/subspace_first_demo/Wbias.npy")), precision)
        spesn.cell.debug_point("xTx", torch.from_numpy(np.load("data/tests/subspace_first_demo/xTx.npy")), precision)
        spesn.cell.debug_point("xTy", torch.from_numpy(np.load("data/tests/subspace_first_demo/xTy.npy")), precision)
        spesn.cell.debug_point("w_ridge_param", 0.0001, precision)
        spesn.cell.debug_point(
            "ridge_xTx",
            torch.from_numpy(np.load("data/tests/subspace_first_demo/ridge_xTx.npy")),
            precision
        )
        spesn.cell.debug_point(
            "inv_xTx",
            torch.from_numpy(np.load("data/tests/subspace_first_demo/inv_xTx.npy")),
            precision
        )
        spesn.cell.debug_point("w", torch.from_numpy(np.load("data/tests/subspace_first_demo/W.npy")), precision)

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
        self.assertAlmostEqual(training_NRMSE, 0.029126309017397423, precision_decimals)

        # No washout this time
        conceptor_net.washout = 0

        # Conceptors ON
        conceptor_net.conceptor_active(True)

        # NRMSE between original and aligned pattern
        NRMSEs_aligned = torch.zeros(4)

        # Train conceptors (Compute C from R)
        conceptors.finalize()

        # Set conceptors in evaluation mode and generate a sample
        for i in range(4):
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
        self.assertAlmostEqual(torch.mean(NRMSEs_aligned).item(), 0.008172502741217613, precision_decimals)

        # Compute similarity matrices
        Rsim_test = conceptors.similarity_matrix(based_on='R')
        Csim_test = conceptors.similarity_matrix(based_on='C')

        # Load similarity matrices
        Rsim = torch.from_numpy(np.load("data/tests/subspace_first_demo/Rsim.npy"))
        Csim = torch.from_numpy(np.load("data/tests/subspace_first_demo/Csim.npy"))

        # Test similarity matrices
        self.assertTensorAlmostEqual(Rsim_test, Rsim, precision)
        self.assertTensorAlmostEqual(Csim_test, Csim, precision)
    # subspace_first_demo

    # endregion PUBLIC

    # region TEST

    # Subspace first demo
    def test_subspace_first_demo(self):
        """
        Subspace first demo
        """
        self.subspace_first_demo(data_dir="subspace_first_demo")
    # end test_subspace_first_demo

    # endregion TEST

# end Test_Subspace_First_Demo
