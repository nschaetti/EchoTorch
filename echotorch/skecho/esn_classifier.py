# -*- coding: utf-8 -*-
#
# File : utils/helpers/ESN.py
# Description : Helper class for ESN classifier
# Date : 27th of April, 2021
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
from typing import Union, List, Dict
import torch
import numpy as np
import echotorch.nn.reservoir
from sklearn.base import ClassifierMixin

# Imports local
from .esn import ESN


# ESN classifier, documentation start
esn_clf_doc_start = """TODO"""

# ESN classifier, additional text
esn_clf_additional_text = """TODO"""

# ESN classifier, additional attribute
esn_clf_additional_attribute = """TODO"""


# ESN Helper class for classification
class ESNClassifier(ESN, ClassifierMixin):
    """ESN helper class for classification

    ESNClassifier description.

    Parameters
    ----------
    input_dim : int
      The size of the input layer.

    reservoir_size : int
      The size of the reservoir recurrent layer.

    leaky_rate : float
        The leaky integrator parameter.

    Attributes
    ----------
    module: torch module (instance)
      The instantiated module.

    """

    # region CONSTRUCTORS

    # Constructors
    def __init__(
            self, input_dim, reservoir_size, *args, leaky_rate=1.0, input_scaling=1.0, nonlin_func=torch.tanh,
            w=None, w_in=None, w_bias=None, learning_algo='inv', ridge_param=0.0, with_bias=True, softmax_output=False,
            washout=0, **kwargs
    ):
        # Create the ESN model
        self.module = echotorch.nn.reservoir.LiESN(
            input_dim=input_dim,
            hidden_dim=reservoir_size,
            leaky_rate=leaky_rate
        )
    # end __init__

    # endregion CONSTRUCTORS

    # region PROPERTIES

    # Default callback
    def _default_callbacks(self):
        """
        TODO
        """
        pass
    # end _default_callbacks

    # endregion PROPERTIES

    # region PUBLIC

    # Initialize and fit the module
    def fit(self, X, y=None, **fit_params):
        """Initialize and fit the module.

        If the module was already initialized, by calling fit, the
        module will be re-initialized (unless ``warm_start`` is True).

        Parameters
        ----------
        X : input data, compatible with echotorch.datasets.EchoDataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * echotorch timetensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        y : target data, compatible with echotorch.datasets.EchoDataset
          The same data types as for ``X`` are supported. If your X is
          a Dataset that contains the target, ``y`` may be set to
          None.

        **fit_params : dict
          @todo

        """
        pass
    # end fit

    # Return the class labels for samples in X.
    def predict(self, X: Union[np.array, torch.tensor, List, echotorch.datasets.EchoDataset, Dict]):
        """Where applicable, return class labels for samples in X.

        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored. If all
        values are relevant, consider using
        :func:`~skorch.NeuralNet.forward` instead.

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        Returns
        -------
        y_pred : numpy ndarray

        """
        pass
    # end predict

    # endregion PUBLIC

    # region PRIVATE

    # endregion PRIVATE

# end ESNClassifier
