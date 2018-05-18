# -*- coding: utf-8 -*-
#

# Imports
import torch
import math
from decimal import Decimal
import numpy as np


# Normalized root-mean-square error
def nrmse(outputs, targets):
    """
    Normalized root-mean square error
    :param outputs: Module's outputs
    :param targets: Target signal to be learned
    :return: Normalized root-mean square deviation
    """
    # Flatten tensors
    outputs = outputs.view(outputs.nelement())
    targets = targets.view(targets.nelement())

    # Check dim
    if outputs.size() != targets.size():
        raise ValueError(u"Ouputs and targets tensors don have the same number of elements")
    # end if

    # Normalization with N-1
    var = torch.std(targets) ** 2

    # Error
    error = (targets - outputs) ** 2

    # Return
    return float(math.sqrt(torch.mean(error) / var))
# end nrmse


# Root-mean square error
def rmse(outputs, targets):
    """
    Root-mean square error
    :param outputs: Module's outputs
    :param targets: Target signal to be learned
    :return: Root-mean square deviation
    """
    # Flatten tensors
    outputs = outputs.view(outputs.nelement())
    targets = targets.view(targets.nelement())

    # Check dim
    if outputs.size() != targets.size():
        raise ValueError(u"Ouputs and targets tensors don have the same number of elements")
    # end if

    # Error
    error = (targets - outputs) ** 2

    # Return
    return float(math.sqrt(torch.mean(error)))
# end rmsd


# Mean square error
def mse(outputs, targets):
    """
    Mean square error
    :param outputs: Module's outputs
    :param targets: Target signal to be learned
    :return: Mean square deviation
    """
    # Flatten tensors
    outputs = outputs.view(outputs.nelement())
    targets = targets.view(targets.nelement())

    # Check dim
    if outputs.size() != targets.size():
        raise ValueError(u"Ouputs and targets tensors don have the same number of elements")
    # end if

    # Error
    error = (targets - outputs) ** 2

    # Return
    return float(torch.mean(error))
# end mse


# Normalized mean square error
def nmse(outputs, targets):
    """
    Normalized mean square error
    :param outputs: Module's output
    :param targets: Target signal to be learned
    :return: Normalized mean square deviation
    """
    # Flatten tensors
    outputs = outputs.view(outputs.nelement())
    targets = targets.view(targets.nelement())

    # Check dim
    if outputs.size() != targets.size():
        raise ValueError(u"Ouputs and targets tensors don have the same number of elements")
    # end if

    # Normalization with N-1
    var = torch.std(targets) ** 2

    # Error
    error = (targets - outputs) ** 2

    # Return
    return float(torch.mean(error) / var)
# end nmse


# Perplexity
def perplexity(output_probs, targets, log=False):
    """
    Perplexity
    :param output_probs: Output probabilities for each word/tokens (length x n_tokens)
    :param targets: Real word index
    :return: Perplexity
    """
    pp = Decimal(1.0)
    e_vec = torch.FloatTensor(output_probs.size(0), output_probs.size(1)).fill_(np.e)
    if log:
        set_p = 1.0 / torch.gather(torch.pow(e_vec, exponent=output_probs.data.cpu()), 1,
                                   targets.data.cpu().unsqueeze(1))
    else:
        set_p = 1.0 / torch.gather(output_probs.data.cpu(), 1, targets.data.cpu().unsqueeze(1))
    # end if
    for j in range(set_p.size(0)):
        pp *= Decimal(set_p[j][0])
    # end for
    return pp
# end perplexity


# Cumulative perplexity
def cumperplexity(output_probs, targets, log=False):
    """
    Cumulative perplexity
    :param output_probs:
    :param targets:
    :param log:
    :return:
    """
    # Get prob of test events
    set_p = torch.gather(output_probs, 1, targets.unsqueeze(1))

    # Make sure it's log
    if not log:
        set_p = torch.log(set_p)
    # end if

    # Log2
    set_log = set_p / np.log(2)

    # sum log
    sum_log = torch.sum(set_log)

    # Return
    return sum_log
# end cumperplexity
