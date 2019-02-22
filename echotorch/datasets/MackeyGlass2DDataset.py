# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset


# Mackey Glass dataset
class MackeyGlass2DDataset(Dataset):
    """
    Mackey Glass 2D dataset
    """

    # Constructor
    def __init__(self, sample_len, n_samples, tau, subsample_rate, normalize=False, seed=None):
        """
        Constructor
        :param sample_len: Length of the time-series in time steps.
        :param n_samples: Number of samples to generate.
        :param tau: Delay of the MG with commonly used value of tau=17 (mild chaos) and tau=30 is moderate chaos.
        :param seed: Seed of random number generator.
        """
        # Properties
        self.sample_len = sample_len
        self.n_samples = n_samples
        self.tau = tau
        self.delta_t = 10
        self.timeseries = 1.2
        self.history_len = tau * self.delta_t
        self.subsample_rate = subsample_rate
        self.normalize = normalize

        # Init seed if needed
        if seed is not None:
            torch.manual_seed(seed)
        # end if
    # end __init__

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
        # Total size
        total_size = self.sample_len
        oldval = 1.2
        samples = list()

        # History
        history = 1.2 * torch.ones(self.history_len) + 0.2 * (torch.rand(self.history_len) - 0.5)

        # For each sample
        for n in range(self.n_samples):
            # Preallocate tensor for time-serie
            sample = torch.zeros(self.sample_len, 2)

            # For each time step
            step = 0
            for t in range(total_size):
                for _ in range(self.delta_t * self.subsample_rate):
                    step = step + 1
                    tauval = history[step % self.history_len]
                    newval = oldval + (0.2 * tauval / (1.0 + tauval**10) - 0.1 * oldval) / self.delta_t
                    history[step % self.history_len] = oldval
                    oldval = newval
                # end for
                sample[t, 0] = newval
                sample[t, 1] = tauval
            # end for

            # Normalize
            if self.normalize:
                maxval = torch.max(sample, dim=0)
                minval = torch.min(sample, dim=0)
                sample = torch.mm(torch.inv(torch.diag(maxval - minval)), (sample - minval.repeat(total_size, 1)))
            # end if

            # Append
            samples.append(sample)
        # end for

        # Squash timeseries through tanh
        return samples
    # end __getitem__

# end MackeyGlassDataset
