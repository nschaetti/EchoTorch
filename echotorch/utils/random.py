# -*- coding: utf-8 -*-
#

# Imports
import random
import numpy as np
import torch


# Set random seed
def manual_seed(seed):
    """
    Set manual seed for pytorch and numpy
    :param seed: Seed for pytorch and numpy
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
# end manual_seed
