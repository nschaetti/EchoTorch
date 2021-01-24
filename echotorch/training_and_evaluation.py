# -*- coding: utf-8 -*-
#
# File : echotorch/training_and_evaluation.py
# Description : Utility functions to easily train and evaluate ESNs
# Date : 23th of January, 2021
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


# Train an ESN with a dataset
def fit(echo_model, dataset):
    """
    Train an ESN with a dataset
    :param echo_model: EchoTorch model to train
    :param dataset: Dataset object to use for training
    """
    pass
# end fit


# Evaluate a trained ESN with a dataset
def eval(echo_model, dataset):
    """
    Evaluate a trained ESN with a dataset
    :param echo_model: EchoTorch model to train
    :param dataset: Dataset object to use for testing
    """
    pass
# end eval


# Train and evaluate an ESN with a dataset using cross-validation
def cross_val_score(echo_model, dataset, eval_function=None, cv=10, n_jobs=None, verbose=False):
    """
    Train and evaluate an ESN with a dataset using cross-validation
    :param echo_model: EchoTorch model to evaluate.
    :param dataset: Dataset to use for training and evaluation.
    :param eval_function: Evaluation function (pytorch loss function)
    :param cv: Cross-validation parameter (default: 10 cross-validation) as integer, or a CrossValidation object.
    :param n_jobs: Number of jobs to run in parallel.
    """
    pass
# end cross_val_score
