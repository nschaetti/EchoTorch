# -*- coding: utf-8 -*-
#

# Imports
import torch
import math


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
