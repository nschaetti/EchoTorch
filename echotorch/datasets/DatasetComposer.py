# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset


# Dataset Composer
class DatasetComposer(Dataset):
    """
    Compose dataset
    """

    # Constructor
    def __init__(self, datasets, *args, **kwargs):
        """
        Constructor
        :param datasets:
        """
        # Super
        super(DatasetComposer, self).__init__(*args, **kwargs)

        # Properties
        self.datasets = datasets
        self.n_datasets = len(datasets)

        # Map item to datasets items
        self.map_items = {}
        self.n_samples = 0
        index = 0
        for i, d in enumerate(datasets):
            for j in range(len(d)):
                self.map_items[index] = (i, j)
                index += 1
                self.n_samples += 1
            # end for
        # end for
    # end __init__

    #region OVERRIDE

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
        (d, e) = self.map_items[idx]
        sample = self.datasets[d][e]
        # print(d, e)
        outputs = self._create_outputs(d, sample.shape[0])
        # print(outputs[:10])
        return self.datasets[d][e], outputs, torch.LongTensor([d])
    # end __getitem__

    #endregion OVERRIDE

    #region PRIVATE

    # Create outputs
    def _create_outputs(self, i, time_length):
        """
        Create outputs
        :param i:
        :param time_length:
        :return:
        """
        # print("create {}".format(i))
        # Create tensor
        outputs = torch.zeros(time_length, self.n_datasets)

        # Put to one
        outputs[:, i] = 1.0

        return outputs
    # end _create_outputs

    #endregion PRIVATE

# end DatasetComposer
