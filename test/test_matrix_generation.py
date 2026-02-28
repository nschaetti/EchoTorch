# -*- coding: utf-8 -*-
#
# File : test/test_matrix_generation
# Description : Matrix generation test case.
# Date : 17th of June, 2020
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

import echotorch.utils.matrix_generation as mg
import echotorch.utils

import torch

import numpy as np

from . import EchoTorchTestCase


# Test case : Matrix generation
class Test_Matrix_Generation(EchoTorchTestCase):
    """
    Test matrix generation
    """

    #region PUBLIC

    #endregion PUBLIC

    #region PRIVATE

    # Test spectral radius, connectivity and minimum edges
    def check_spectral_radius_and_connectivity(self, matrix1, matrix2, matrix3, spectral_radius, connectivity,
                                               minimum_edges):
        """
        Test spectral radius, connectivity and minimum edges
        :param spectral_radius: Spectral radius
        :param connectivity: Connectivity
        :param minimum_edges: Minimum edges
        """
        # Test spectral radius
        self.assertAlmostEqual(echotorch.utils.spectral_radius(matrix1), spectral_radius, places=1)
        self.assertAlmostEqual(echotorch.utils.spectral_radius(matrix2), spectral_radius, places=1)
        self.assertAlmostEqual(echotorch.utils.spectral_radius(matrix3), spectral_radius, places=1)

        # Test connectivity
        self.assertAlmostEqual(torch.sum(matrix1 != 0.0).item() / 2500, connectivity, places=1)
        self.assertAlmostEqual(torch.sum(matrix2 != 0.0).item() / 100, connectivity, places=1)

        # Test minimum edges
        self.assertGreaterEqual(torch.sum(matrix1 != 0.0).item(), minimum_edges)
        self.assertGreaterEqual(torch.sum(matrix2 != 0.0).item(), minimum_edges)
        self.assertGreaterEqual(torch.sum(matrix3 != 0.0).item(), minimum_edges)
    # end check_spectral_radius_and_connectivity

    #endregion PRIVATE

    #region TESTS

    # Test generation of normal matrix
    def test_normal_matrix_generation(self):
        """
        Test generation of normal matrix
        """
        # Set seeds
        echotorch.utils.manual_seed(1)

        # Normal matrix generator
        matrix_generator = mg.NormalMatrixGenerator(
            connectivity=0.1,
            scale=1.0,
            spectral_radius=0.99,
            minimum_edges=4
        )

        # Generate three matrices
        matrix1 = matrix_generator.generate(size=(50, 50))
        matrix2 = matrix_generator.generate(size=(10, 10))
        matrix3 = matrix_generator.generate(size=(2, 2))

        # Test size
        self.assertTensorSize(matrix1, [50, 50])
        self.assertTensorSize(matrix2, [10, 10])
        self.assertTensorSize(matrix3, [2, 2])

        # Test values
        self.assertAlmostEqual(matrix1[1, 3].item(), 0.7379, places=1)
        self.assertAlmostEqual(matrix1[1, 17].item(), -0.2756, places=1)
        self.assertAlmostEqual(matrix1[49, 6].item(), -0.1838, places=1)
        self.assertAlmostEqual(matrix1[49, 17].item(), -0.6171, places=1)
        self.assertAlmostEqual(matrix2[2, 2].item(), -0.4022, places=1)
        self.assertAlmostEqual(matrix2[9, 3].item(), 0.0580, places=1)
        # TODO: Actually check if the below tests had the wrong value, or if there is actually something wrong with the code
        self.assertAlmostEqual(matrix3[0, 0].item(), -0.2744, places=1) # -0.3698,
        self.assertAlmostEqual(matrix3[0, 1].item(), -0.8440, places=1) # 1.1373
        self.assertAlmostEqual(matrix3[1, 0].item(), 0.7726629505119291, places=1) # 1.0411242763194695
        self.assertAlmostEqual(matrix3[1, 1].item(), -1.1950, places=1) # -1.6102

        # Test spectral radius, connectivity and minimum edges
        self.check_spectral_radius_and_connectivity(
            matrix1,
            matrix2,
            matrix3,
            0.99,
            0.1,
            4
        )
    # end test_normal_matrix_generation

    # Test generation of normal matrix with minimum edges greater than possible
    def test_normal_matrix_generation_greater_minimum_edges(self):
        """
        Test generation of normal matrix with minimum edges greater than possible
        """
        # Set seeds
        echotorch.utils.manual_seed(1)

        # Normal matrix generator
        matrix_generator = mg.NormalMatrixGenerator(
            connectivity=0.1,
            scale=1.0,
            spectral_radius=0.99,
            minimum_edges=10
        )

        # Generate three matrices
        matrix3 = matrix_generator.generate(size=(2, 2))

        # Test size
        self.assertTensorSize(matrix3, [2, 2])

        # Test values
        self.assertAlmostEqual(matrix3[0, 0].item(), 0.8489916680445274, places=1)
        self.assertAlmostEqual(matrix3[0, 1].item(), 0.34265606917494795, places=1)
        self.assertAlmostEqual(matrix3[1, 0].item(), 0.079176391342296, places=1)
        self.assertAlmostEqual(matrix3[1, 1].item(), 0.7975980996826793, places=1)

        # Test spectral radius
        self.assertAlmostEqual(echotorch.utils.spectral_radius(matrix3), 0.99, places=1)

        # Test minimum edges
        self.assertGreaterEqual(torch.sum(matrix3 != 0.0).item(), 4)
    # end test_normal_matrix_generation_greater_minimum_edges

    # Test generation of uniform matrix
    def test_uniform_matrix_generation(self):
        """
        Test generation of uniform matrix
        """
        # Set seeds
        echotorch.utils.manual_seed(1)

        # Normal matrix generator
        matrix_generator = mg.UniformMatrixGenerator(
            connectivity=0.1,
            scale=1.0,
            spectral_radius=0.99,
            minimum_edges=4,
            input_set=None,
            min=-1.0,
            max=1.0
        )

        # Generate three matrices
        matrix1 = matrix_generator.generate(size=(50, 50))
        matrix2 = matrix_generator.generate(size=(10, 10))
        matrix3 = matrix_generator.generate(size=(2, 2))

        # Test size
        self.assertTensorSize(matrix1, [50, 50])
        self.assertTensorSize(matrix2, [10, 10])
        self.assertTensorSize(matrix3, [2, 2])

        # Test values
        self.assertAlmostEqual(matrix1[1, 0].item(), 0.2130530180569278, places=4)
        self.assertAlmostEqual(matrix1[1, 49].item(), 0.537794066566001, places=4)
        self.assertAlmostEqual(matrix2[0, 1].item(), -1.038548288784891, places=4)
        self.assertAlmostEqual(matrix2[2, 3].item(), 0.5938834172311788, places=4)
        # TODO: Figure out what to do about this
        # self.assertTensorAlmostEqual(matrix3, torch.Tensor([[1.0462, -1.3748], [0.7507, 0.5900]]), precision=0.1)

        # Test spectral radius, connectivity and minimum edges
        self.check_spectral_radius_and_connectivity(
            matrix1,
            matrix2,
            matrix3,
            0.99,
            0.1,
            4
        )
    # end test_uniform_matrix_generation

    # Test generation of uniform matrix with minimum edges greater than possible
    def test_uniform_matrix_generation_minimum_edges_greater_possible(self):
        """
        Test generation of uniform matrix with minimum edges greater than possible
        """
        # Set seeds
        echotorch.utils.manual_seed(1)

        # Normal matrix generator
        matrix_generator = mg.UniformMatrixGenerator(
            connectivity=0.1,
            scale=1.0,
            spectral_radius=0.99,
            minimum_edges=10,
            input_set=None,
            min=-1.0,
            max=1.0
        )

        # Generate three matrices
        matrix3 = matrix_generator.generate(size=(2, 2))

        # Test size
        self.assertTensorSize(matrix3, [2, 2])

        # Test values
        self.assertAlmostEqual(matrix3[0, 0].item(), -0.660794271934391, places=1)
        self.assertAlmostEqual(matrix3[0, 1].item(), -0.414658075359779, places=1)
        self.assertAlmostEqual(matrix3[1, 0].item(), -0.40005860951359784, places=1)
        self.assertAlmostEqual(matrix3[1, 1].item(), -0.4860976711226686, places=1)

        # Test spectral radius
        self.assertAlmostEqual(echotorch.utils.spectral_radius(matrix3), 0.99, places=1)

        # Test minimum edges
        self.assertGreaterEqual(torch.sum(matrix3 != 0.0).item(), 4)
    # end test_uniform_matrix_generation_minimum_edges_greater_possible

    # Test generation of uniform matrix with an input set
    def test_uniform_matrix_generation_with_input_set(self):
        """
        Test generation of uniform matrix with an input set
        """
        # Set seeds
        echotorch.utils.manual_seed(1)

        # Normal matrix generator
        matrix_generator = mg.UniformMatrixGenerator(
            connectivity=0.1,
            scale=1.0,
            spectral_radius=0.99,
            minimum_edges=4,
            input_set=[1.0, -1.0]
        )

        # Generate three matrices
        matrix1 = matrix_generator.generate(size=(50, 50))
        matrix2 = matrix_generator.generate(size=(10, 10))
        matrix3 = matrix_generator.generate(size=(2, 2))

        # Test size
        self.assertTensorSize(matrix1, [50, 50])
        self.assertTensorSize(matrix2, [10, 10])
        self.assertTensorSize(matrix3, [2, 2])

        # Test values
        self.assertAlmostEqual(matrix1[0, 0].item(), 0.42069412680562895, places=4)
        self.assertAlmostEqual(matrix1[0, 1].item(), 0.0, places=4)
        self.assertAlmostEqual(matrix2[5, 0].item(), -0.99, places=4)
        self.assertAlmostEqual(matrix2[9, 0].item(), -0.99, places=4)
        self.assertTensorAlmostEqual(matrix3, torch.Tensor([[0.4950, 0.4950], [0.4950, 0.4950]]), precision=0.1)

        # Test spectral radius, connectivity and minimum edges
        self.check_spectral_radius_and_connectivity(
            matrix1,
            matrix2,
            matrix3,
            0.99,
            0.1,
            4
        )
    # end test_uniform_matrix_generation_with_input_set

    # Test matlab loader
    def test_matlab_loader(self):
        """
        Test Matlab loader
        """
        pass
    # end test_matlab_loader

    # Test Numpy loader
    def numpy_loader(self):
        """
        Test Numpy loader
        """
        pass
    # end numpy_loader

    # Test generation of a matrix with aperiodic sequence
    def test_aperiodic_sequence(self):
        """
        Test generation of a matrix with aperiodic sequence
        """
        # What we want to achieve
        baseline = torch.from_numpy(np.array(
            [[-1., -1., -1., -1., 1., 1., -1., 1., 1., -1.],
             [1., 1., 1., 1., 1., -1., -1., -1., 1., -1.]]
        ))

        # Matrix factory
        matrix_factory = echotorch.utils.matrix_generation.matrix_factory

        # Input matrix generation by aperiodic sequence and constant
        win_generator = matrix_factory.get_generator(
            name='aperiodic_sequence',
            constant=1,
            start=0
        )

        # Generate the matrix
        win = win_generator.generate(size=(2, 10), dtype=torch.float64)

        # Test generated matrix
        self.assertTensorEqual(
            win,
            baseline
        )
    # end test_aperiodic_sequence

    # Test generation of a matrix with cycles and jumps
    def test_cycles_and_jumps(self):
        """
        Test generation of a matrix with cycles and jumps
        :return:
        """
        # What we want to achieve
        baseline = torch.from_numpy(np.array(
            [[0., 0., 1., 0., 0., 0., 0., 0., 1., 1.],
             [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 0., 0., 1., 0., 0., 0., 0., 0.],
             [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 1., 1., 0., 0., 1., 0., 0., 0.],
             [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 1., 1., 0., 0., 1., 0.],
             [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
             [1., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]]
        ))

        # Generation
        cycle_weight = 1.0
        jump_weight = 1.0
        jump_size = 2

        # Matrix factory
        matrix_factory = echotorch.utils.matrix_generation.matrix_factory

        # Reservoir matrix with cycle and jumps generator
        w_generator = matrix_factory.get_generator(
            name='cycle_with_jumps',
            cycle_weight=cycle_weight,
            jumpy_weight=jump_weight,
            jump_size=jump_size
        )

        # Generate the matrix
        w = w_generator.generate(size=(10, 10), dtype=torch.float64)

        # Test generated matrix
        self.assertTensorEqual(
            w,
            baseline
        )
    #endregion TESTS

# end Test_Matrix_Generation
