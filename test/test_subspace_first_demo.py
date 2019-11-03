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
import unittest
from unittest import TestCase

from echotorch.datasets.NARMADataset import NARMADataset
import echotorch.nn.reservoir as etrs
import echotorch.nn.conceptors as etnc
import echotorch.utils.matrix_generation as mg
import echotorch.utils
import echotorch.datasets as etds
from echotorch.datasets import DatasetComposer

import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

import numpy as np

from echotorch.nn.Node import Node
from .EchoTorchTestCase import EchoTorchTestCase


# Test case : Subspace first demo.
class Test_Subspace_First_Demo(EchoTorchTestCase):
    """
    Test subspace first demo
    """

    ##############################
    # PUBLIC
    ##############################

    # Subspace first demo
    def subspace_first_demo(self, data_dir, reservoir_size=100, spectral_radius=1.5, input_scaling=1.5, bias_scaling=0.2,
                            connectivity=10.0, washout_length=500, learn_length=1000, ridge_param_wstar=0.0001,
                            ridge_param_wout=0.01, aperture=10, precision=0.000001, torch_seed=1, np_seed=1):
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

        # Init random number generators
        torch.random.manual_seed(torch_seed)
        np.random.seed(np_seed)

        # Parameters
        connectivity = connectivity / reservoir_size
        signal_plot_length = 20
        conceptor_test_length = 200
        singular_plot_length = 50
        free_run_length = 100000
        apertures = [1.0, 10.0, 100.0, 1000.0, 10000.0]
        dtype = torch.float64

        # Load internal weights W*
        wstar_generator = mg.matrix_factory.get_generator(
            "matlab",
            file_name=os.path.join(TEST_PATH, "WstarRaw.mat"),
            entity_name="WstarRaw"
        )

        # Load input-internal weights
        win_generator = mg.matrix_factory.get_generator(
            "matlab",
            file_name=os.path.join(TEST_PATH, "WinRaw.mat"),
            entity_name="WinRaw"
        )

        # Load bias
        wbias_generator = mg.matrix_factory.get_generator(
            "matlab",
            file_name=os.path.join(TEST_PATH, "Wbias.mat"),
            entity_name="Wbias",
            shape=reservoir_size
        )

        # First sine pattern
        pattern1_training = etds.SinusoidalTimeseries(
            sample_len=washout_length + learn_length,
            n_samples=1,
            a=1,
            period=8.8342522,
            dtype=dtype
        )

        # Second sine pattern
        pattern2_training = etds.SinusoidalTimeseries(
            sample_len=washout_length + learn_length,
            n_samples=1,
            a=1,
            period=9.8342522,
            dtype=dtype
        )

        # First periodic pattern
        pattern3_training = etds.PeriodicSignalDataset(
            sample_len=washout_length + learn_length,
            n_samples=1,
            period=[0.9000000000000002, -0.11507714997817164, 0.17591170369788622, -0.9, -0.021065045054201592],
            dtype=dtype
        )

        # Second periodic pattern
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
        spesn = etnc.SPESN(
            input_dim=1,
            hidden_dim=reservoir_size,
            output_dim=1,
            spectral_radius=spectral_radius,
            learning_algo='inv',
            w_generator=wstar_generator,
            win_generator=win_generator,
            wbias_generator=wbias_generator,
            input_scaling=input_scaling,
            bias_scaling=bias_scaling,
            ridge_param=ridge_param_wout,
            w_ridge_param=ridge_param_wstar,
            washout=washout_length,
            debug=debug_mode,
            test_case=self,
            dtype=dtype
        )

        # Load sample matrices
        for i in range(4):
            # Input patterns
            spesn.cell.debug_point(
                "u{}".format(i),
                torch.reshape(torch.from_numpy(np.load(os.path.join(TEST_PATH, "u{}.npy".format(i)))), shape=(-1, 1)),
                precision
            )

            # States
            spesn.cell.debug_point(
                "X{}".format(i),
                torch.from_numpy(np.load(os.path.join(TEST_PATH, "X{}.npy".format(i)))),
                precision
            )

            # Targets
            spesn.cell.debug_point(
                "Y{}".format(i),
                torch.from_numpy(np.load(os.path.join(TEST_PATH, "Y{}.npy".format(i)))),
                precision
            )

            # Xold
            spesn.cell.debug_point(
                "Xold{}".format(i),
                torch.from_numpy(np.load(os.path.join(TEST_PATH, "Xold{}.npy".format(i)))),
                precision
            )
        # end for

        # Load debug W, xTx, xTy
        spesn.cell.debug_point(
            "Wstar",
            torch.from_numpy(np.load(os.path.join(TEST_PATH, "Wstar.npy"), allow_pickle=True)),
            precision
        )
        spesn.cell.debug_point("Win", torch.from_numpy(np.load(os.path.join(TEST_PATH, "Win.npy"))), precision)
        spesn.cell.debug_point("Wbias", torch.from_numpy(np.load(os.path.join(TEST_PATH, "Wbias.npy"))), precision)
        spesn.cell.debug_point("xTx", torch.from_numpy(np.load(os.path.join(TEST_PATH, "xTx.npy"))), precision)
        spesn.cell.debug_point("xTy", torch.from_numpy(np.load(os.path.join(TEST_PATH, "xTy.npy"))), precision)
        spesn.cell.debug_point("w_ridge_param", 0.0001, precision)
        spesn.cell.debug_point(
            "ridge_xTx",
            torch.from_numpy(np.load(os.path.join(TEST_PATH, "ridge_xTx.npy"))),
            precision
        )
        spesn.cell.debug_point("inv_xTx",  torch.from_numpy(np.load(os.path.join(TEST_PATH, "inv_xTx.npy"))), 0.001)
        spesn.cell.debug_point("w", torch.from_numpy(np.load(os.path.join(TEST_PATH, "W.npy"))), 0.00001)

        # Xold and Y collectors
        Xold_collector = torch.empty(4 * learn_length, reservoir_size, dtype=dtype)
        Y_collector = torch.empty(4 * learn_length, reservoir_size, dtype=dtype)

        # Go through dataset
        for i, data in enumerate(patterns_loader):
            # Inputs and labels
            inputs, outputs, labels = data

            # Feed SP-ESN
            X = spesn(inputs, inputs)

            # Get targets
            Y = spesn.cell.targets(X[0])

            # Get features
            Xold = spesn.cell.features(X[0])

            # Save
            Xold_collector[i * learn_length:i * learn_length + learn_length] = Xold
            Y_collector[i * learn_length:i * learn_length + learn_length] = Y
        # end for

        # Finalize training
        spesn.finalize()

        # Predicted by W
        predY = torch.mm(spesn.cell.w, Xold_collector.t()).t()

        # Compute and test NRMSE
        training_NRMSE = echotorch.utils.nrmse(predY, Y_collector)
        self.assertAlmostEqual(training_NRMSE, 0.029126309017399498, places=4)
        print("DEBUG - INFO: Training NRMSE : {}".format(training_NRMSE))

        # Run trained ESN with empty inputs
        generated = spesn(torch.zeros(1, 2000, 1, dtype=dtype))
    # subspace_first_demo

    ##############################
    # TESTS
    ##############################

    # Subspace first demo
    def test_subspace_first_demo(self):
        """
        Subspace first demo
        """
        self.subspace_first_demo(data_dir="subspace_first_demo")
    # end test_subspace_first_demo

# end Test_Subspace_First_Demo
