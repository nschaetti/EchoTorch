# -*- coding: utf-8 -*-
#
# File : test/test_latch_copy_repeat_tasks.py
# Description : Test dataset generation for latch, copy and repeat tasks
# Date : 16th of July, 2020
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
import echotorch.utils
from . import EchoTorchTestCase
import numpy as np
import torch
import echotorch.datasets as etda
import echotorch.utils
from torch.utils.data.dataloader import DataLoader


# Test case : Test dataset generation for latch, copy and repeat tasks
class Test_Latch_Copy_Repeat_Tasks(EchoTorchTestCase):
    """
    Test case : Test dataset generation for latch, copy and repeat tasks
    """
    # region BODY

    # region PUBLIC

    # endregion PUBLIC

    # region TEST

    # Test latch task generation
    def test_latch_task(self):
        """
        Test latch task generation
        """
        # Init. random number generators
        echotorch.utils.manual_seed(1)

        # Inputs baseline
        inputs_baseline = torch.from_numpy(np.array(
            [[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
             0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0.]]
        ))

        # Outputs baseline
        outputs_baseline = torch.from_numpy(np.array(
            [[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0.,
             0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1., 1., 1., 1., 1.]]
        ))

        # Latch task dataset
        latch_task_dataset = etda.LatchTaskDataset(
            n_samples=1,
            length_min=40,
            length_max=50,
            n_pics=3,
            dtype=torch.float64
        )

        # Dataset loader
        latch_task_loader = torch.utils.data.DataLoader(
            latch_task_dataset,
            batch_size=1,
            shuffle=False
        )

        # Get inputs and output
        task_inputs, task_outputs = next(iter(latch_task_loader))

        # Compare to input baseline
        self.assertTensorEqual(inputs_baseline, task_inputs[0])

        # Compare to output baseline
        self.assertTensorEqual(outputs_baseline, task_outputs[0])
    # end test_latch_task

    # Test copy task
    def test_copy_task(self):
        """
        Test copy task
        """
        # Init. random number generators
        echotorch.utils.manual_seed(1)

        # Inputs baseline
        inputs_baseline = torch.from_numpy(np.array(
            [[1., 0., 0., 1., 1., 1., 1., 1., 0.],
             [0., 0., 1., 0., 1., 1., 0., 0., 0.],
             [1., 0., 0., 0., 1., 0., 0., 1., 0.],
             [0., 0., 0., 1., 0., 0., 0., 1., 0.],
             [1., 1., 1., 1., 0., 0., 0., 1., 0.],
             [1., 1., 1., 1., 1., 0., 1., 1., 0.],
             [0., 0., 1., 0., 0., 1., 1., 1., 0.],
             [0., 1., 0., 0., 1., 1., 0., 1., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 1.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.]]
        ))

        # Outputs baseline
        outputs_baseline = torch.from_numpy(np.array(
            [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 0., 0., 1., 1., 1., 1., 1., 0.],
             [0., 0., 1., 0., 1., 1., 0., 0., 0.],
             [1., 0., 0., 0., 1., 0., 0., 1., 0.],
             [0., 0., 0., 1., 0., 0., 0., 1., 0.],
             [1., 1., 1., 1., 0., 0., 0., 1., 0.],
             [1., 1., 1., 1., 1., 0., 1., 1., 0.],
             [0., 0., 1., 0., 0., 1., 1., 1., 0.],
             [0., 1., 0., 0., 1., 1., 0., 1., 0.]]

        ))

        # Copy task dataset
        copy_task_dataset = etda.CopyTaskDataset(
            n_samples=10,
            length_min=1,
            length_max=20,
            n_inputs=8,
            dtype=torch.float64
        )

        # Dataset loader
        copy_task_loader = torch.utils.data.DataLoader(
            copy_task_dataset,
            batch_size=1,
            shuffle=False
        )

        # Get inputs and output
        task_inputs, task_outputs = next(iter(copy_task_loader))

        # Compare to input baseline
        self.assertTensorEqual(inputs_baseline, task_inputs[0])

        # Compare to output baseline
        self.assertTensorEqual(outputs_baseline, task_outputs[0])
    # end test_copy_task

    # Test repeat task
    def test_repeat_task(self):
        """
        Test repeat task
        """
        # Init. random number generators
        echotorch.utils.manual_seed(1)

        # Inputs baseline
        inputs_baseline = torch.from_numpy(np.array(
            [[[0., 0., 1., 1., 1., 1., 1., 0., 0.],
              [0., 1., 0., 1., 1., 0., 0., 1., 0.],
              [0., 0., 0., 1., 0., 0., 1., 0., 0.],
              [0., 0., 1., 0., 0., 0., 1., 1., 0.],
              [1., 1., 1., 0., 0., 0., 1., 1., 0.],
              [1., 1., 1., 1., 0., 1., 1., 0., 0.],
              [0., 1., 0., 0., 1., 1., 1., 0., 0.],
              [1., 0., 0., 1., 1., 0., 1., 1., 0.],
              [1., 1., 0., 0., 1., 1., 0., 0., 0.],
              [0., 0., 1., 1., 1., 0., 1., 0., 0.],
              [0., 1., 1., 0., 1., 1., 0., 1., 0.],
              [0., 0., 1., 1., 1., 0., 1., 1., 0.],
              [0., 1., 1., 1., 1., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 1.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.]]]
        ))

        # Outputs baseline
        outputs_baseline = torch.from_numpy(np.array(
            [[[0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 1., 1., 1., 1., 1., 0., 0.],
              [0., 1., 0., 1., 1., 0., 0., 1., 0.],
              [0., 0., 0., 1., 0., 0., 1., 0., 0.],
              [0., 0., 1., 0., 0., 0., 1., 1., 0.],
              [1., 1., 1., 0., 0., 0., 1., 1., 0.],
              [1., 1., 1., 1., 0., 1., 1., 0., 0.],
              [0., 1., 0., 0., 1., 1., 1., 0., 0.],
              [1., 0., 0., 1., 1., 0., 1., 1., 0.],
              [1., 1., 0., 0., 1., 1., 0., 0., 0.],
              [0., 0., 1., 1., 1., 0., 1., 0., 0.],
              [0., 1., 1., 0., 1., 1., 0., 1., 0.],
              [0., 0., 1., 1., 1., 0., 1., 1., 0.],
              [0., 1., 1., 1., 1., 0., 0., 0., 0.],
              [0., 0., 1., 1., 1., 1., 1., 0., 0.],
              [0., 1., 0., 1., 1., 0., 0., 1., 0.],
              [0., 0., 0., 1., 0., 0., 1., 0., 0.],
              [0., 0., 1., 0., 0., 0., 1., 1., 0.],
              [1., 1., 1., 0., 0., 0., 1., 1., 0.],
              [1., 1., 1., 1., 0., 1., 1., 0., 0.],
              [0., 1., 0., 0., 1., 1., 1., 0., 0.],
              [1., 0., 0., 1., 1., 0., 1., 1., 0.],
              [1., 1., 0., 0., 1., 1., 0., 0., 0.],
              [0., 0., 1., 1., 1., 0., 1., 0., 0.],
              [0., 1., 1., 0., 1., 1., 0., 1., 0.],
              [0., 0., 1., 1., 1., 0., 1., 1., 0.],
              [0., 1., 1., 1., 1., 0., 0., 0., 0.]]]
        ))

        # Repeat task dataset
        repeat_task_dataset = etda.RepeatTaskDataset(
            n_samples=1,
            length_min=1,
            length_max=20,
            n_inputs=8,
            max_repeat=2,
            dtype=torch.float64
        )

        # Dataset loader
        repeat_task_loader = torch.utils.data.DataLoader(
            repeat_task_dataset,
            batch_size=1,
            shuffle=False
        )

        # Get inputs and output
        task_inputs, task_outputs = next(iter(repeat_task_loader))

        # Compare to input baseline
        self.assertTensorEqual(inputs_baseline, task_inputs)

        # Compare to output baseline
        self.assertTensorEqual(outputs_baseline, task_outputs)
    # end test_copy_task

    # endregion TEST

    # endregion BODY
# end Test_Latch_Copy_Repeat_Tasks
