
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import echotorch.nn
import echotorch.datasets

# Reuters C50 dataset
c50_dataset = echotorch.datasets.ReutersC50Dataset(root="./data", download=False)
trainloader = torch.utils.data.DataLoader(c50_dataset, batch_size=4)

# For each batch
for i, data in enumerate(trainloader):
    # Inputs and labels
    inputs, labels = data
# end for
