
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import echotorch.nn


if __name__ == "__main__":

    batch_size = 4

    # Variable
    u = Variable(torch.rand(batch_size, 2))

    # Initial state
    initial_state = Variable(torch.zeros(5), requires_grad=False)

    # ESN
    esn = echotorch.nn.Reservoir(2, 2, batch_size, bias=False)
    p, x = esn(u, initial_state)
    print(x)
    print(p)

# end if