
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import echotorch.nn


if __name__ == "__main__":

    batch_size = 10

    # Variable
    u = Variable(torch.rand(10, 2))

    initial_states = Variable(torch.zeros(batch_size, 100), requires_grad=False)

    # ESN
    esn = echotorch.nn.Reservoir(2, 2, 100, bias=False)
    p, x = esn(u, initial_states)
    print(x)
    print(p)

# end if