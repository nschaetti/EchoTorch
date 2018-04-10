# -*- coding: utf-8 -*-
#

# Imports
import echotorch.datasets
from torch.utils.data.dataloader import DataLoader


# Logistoc map dataset
log_map = echotorch.datasets.LogisticMapDataset(10000, 10)

# Dataset
log_map_dataset = DataLoader(log_map, batch_size=10, shuffle=True)

# For each sample
for data in log_map_dataset:
    print(data[0])
# end for
