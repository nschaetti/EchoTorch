
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import echotorch.nn


if __name__ == "__main__":

    # Variable
    u = Variable(torch.rand(10, 2))

    # ESN
    esn = echotorch.nn.Reservoir(2, 2, 100, bias=False)
    esn(u)

# end if