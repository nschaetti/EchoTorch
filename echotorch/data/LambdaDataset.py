# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset

# Local imports
from .EchoDataset import EchoDataset


# Lambda dataset
class LambdaDataset(EchoDataset):
    """
    Create simple periodic signal timeseries
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, sample_len, n_samples, func, start=0, dtype=None):
        """
        Constructor
        :param sample_len: Sample's length
        :param period:
        """
        # Properties
        self.sample_len = sample_len
        self.n_samples = n_samples
        self.func = func
        self.start = start
        self.dtype = dtype

        # Generate data set
        self.outputs = self._generate()
    # end __init__

    # endregion CONSTRUCTORS

    # region PRIVATE

    # Generate
    def _generate(self):
        """
        Generate dataset
        :return:
        """
        # List of samples
        samples = list()

        # For each sample
        for i in range(self.n_samples):
            # Tensor
            sample = torch.zeros(self.sample_len, 1, dtype=self.dtype)

            # Timestep
            for t in range(self.sample_len):
                sample[t, 0] = self.func(t + self.start)
            # end for

            # Append
            samples.append(sample)
        # end for

        return samples
    # end _generate

    # endregion PRIVATE

    # region OVERRIDE

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return self.n_samples
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        return self.outputs[idx]
    # end __getitem__

    # endregion OVERRIDE

# end LambdaDataset
