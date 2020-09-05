# -*- coding: utf-8 -*-
#
# File : test/test_boolean_operations.py
# Description : Test boolean operation on Conceptors
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
from . import EchoTorchTestCase
import numpy as np
import torch
import echotorch.nn.conceptors as ecnc
import echotorch.utils.matrix_generation as mg


# Test case : boolean operations on Conceptors
class Test_Boolean_Operations(EchoTorchTestCase):
    """
    boolean operations on Conceptors
    """
    # region BODY

    # region PUBLIC

    # Boolean operations
    def boolean_operations(self, data_dir, exA, exB, exAandB, exAorB, exnotA, reservoir_size=2,
                           precision=0.001, torch_seed=1, np_seed=1, use_matlab_params=True):
        """
        Memory management
        """
        # Package
        subpackage_dir, this_filename = os.path.split(__file__)
        package_dir = os.path.join(subpackage_dir, "..")
        TEST_PATH = os.path.join(package_dir, "data", "tests", data_dir)

        # Random numb. init
        torch.random.manual_seed(torch_seed)
        np.random.seed(np_seed)

        # Type params
        dtype = torch.float64

        # Load X state matrix from file or from random distrib ?
        if use_matlab_params:
            # Load state matrix X
            x_generator = mg.matrix_factory.get_generator(
                "matlab",
                file_name=os.path.join(TEST_PATH, "X"),
                entity_name="X"
            )

            # Load state matrix Y
            y_generator = mg.matrix_factory.get_generator(
                "matlab",
                file_name=os.path.join(TEST_PATH, "Y"),
                entity_name="Y"
            )
        else:
            # Generate internal weights
            x_generator = mg.matrix_factory.get_generator(
                "normal",
                mean=0.0,
                std=1.0
            )

            # Generate internal weights
            y_generator = mg.matrix_factory.get_generator(
                "normal",
                mean=0.0,
                std=1.0
            )
        # end if

        # Generate X and Y
        X = x_generator.generate(size=(reservoir_size, reservoir_size), dtype=dtype)
        Y = y_generator.generate(size=(reservoir_size, reservoir_size), dtype=dtype)

        # Transpose on time dim / reservoir dim
        X = X.t()
        Y = Y.t()

        # Add batch dimension
        X = X.reshape(1, reservoir_size, reservoir_size)
        Y = Y.reshape(1, reservoir_size, reservoir_size)

        # Create a conceptor
        A = ecnc.Conceptor(input_dim=reservoir_size, aperture=1, dtype=dtype)

        # Learn from state
        A.filter_fit(X)

        # Divide correlation matrix R by reservoir dimension
        # and update C.
        Ra = A.correlation_matrix()
        A.set_R(Ra)

        # Get conceptor matrix
        Ua, Sa, Va = A.SVD

        # Change singular values
        Sa[0] = 0.95
        Sa[1] = 0.2

        # New C
        Cnew = torch.mm(torch.mm(Ua, torch.diag(Sa)), Va)

        # Recompute conceptor
        A.set_C(Cnew, aperture=1)

        # Create a conceptor
        B = ecnc.Conceptor(input_dim=reservoir_size, aperture=1, dtype=dtype)

        # Learn from state
        B.filter_fit(Y)

        # Divide correlation matrix R by reservoir dimension
        # and update C.
        Rb = B.correlation_matrix()
        B.set_R(Rb)

        # Get conceptor matrix
        Ub, Sb, Vb = B.SVD

        # Change singular values
        Sb[0] = 0.8
        Sb[1] = 0.3

        # Recompute conceptor
        B.set_C(torch.mm(torch.mm(Ub, torch.diag(Sb)), Vb), aperture=1)

        # AND, OR, NOT
        AandB = ecnc.Conceptor.operator_AND(A, B)
        AorB = ecnc.Conceptor.operator_OR(A, B)
        notA = ecnc.Conceptor.operator_NOT(A)

        # Test conceptor matrix
        self.assertArrayAlmostEqual(A.conceptor_matrix().numpy(), exA, precision)
        self.assertArrayAlmostEqual(B.conceptor_matrix().numpy(), exB, precision)
        self.assertArrayAlmostEqual(AandB.conceptor_matrix().numpy(), exAandB, precision)
        self.assertArrayAlmostEqual(AorB.conceptor_matrix().numpy(), exAorB, precision)
        self.assertArrayAlmostEqual(notA.conceptor_matrix().numpy(), exnotA, precision)
    # end boolean_operations

    # endregion PUBLIC

    # region TEST

    # Boolean operations with matlab params
    def test_memory_management_matlab(self):
        """
        Boolean operations with matlab params
        """
        # Test with matlab params
        self.boolean_operations(
            data_dir="boolean_operations",
            use_matlab_params=True,
            exA=np.array([[ 0.7503, -0.3315], [-0.3315,  0.3997]]),
            exB=np.array([[ 0.7766, -0.1057], [-0.1057,  0.3234]]),
            exAandB=np.array([[ 0.5955, -0.2104], [-0.2104,  0.2360]]),
            exAorB=np.array([[ 0.8547, -0.1921], [-0.1921,  0.5997]]),
            exnotA=np.array([[0.2497, 0.3315], [0.3315, 0.6003]])
        )
    # end test_memory_management_matlab

    # Boolean operations with random matrix
    def test_memory_management(self):
        """
        Boolean operations with matlab params
        """
        # Test with matlab params
        self.boolean_operations(
            data_dir="boolean_operations",
            use_matlab_params=False,
            torch_seed=1,
            np_seed=1,
            exA=np.array([[0.6786, 0.3604], [0.3604, 0.4714]]),
            exB=np.array([[0.3795, 0.1829], [0.1829, 0.7205]]),
            exAandB=np.array([[0.3596, 0.2302], [0.2302, 0.3882]]),
            exAorB=np.array([[0.7972, 0.1825], [0.1825, 0.7486]]),
            exnotA=np.array([[0.3214, -0.3604], [-0.3604,  0.5286]])
        )
    # end test_memory_management

    # endregion TEST

    # endregion BODY
# end Test_Memory_Management
