# -*- coding: utf-8 -*-
#

# Imports
import torch


# Normalized root-mean-square deviation
def nrmse(outputs, targets):
    """
    Normalized root-mean square deviation
    :param outputs:
    :param targets:
    :return:
    """
    mean = torch.Tensor(1).fill_(torch.mean(torch.pow(outputs - targets, 2)))
    return torch.sqrt(mean) / (torch.max(targets) - torch.min(targets))
# end nrmse
