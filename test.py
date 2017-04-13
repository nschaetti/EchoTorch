
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import echotorch.nn


if __name__ == "__main__":

    batch_size = 4
    reservoir_size = 5

    # Variable
    u = Variable((torch.rand(batch_size, 2) - 0.5) * 2.0)
    print("u : ")
    print(u)
    # Initial state
    initial_state = Variable(torch.zeros(reservoir_size), requires_grad=False)

    # ESN
    esn = echotorch.nn.Reservoir(2, 2, reservoir_size, bias=False)
    p, x = esn(u, initial_state)
    print("X : ")
    print(x)
    print("P : ")
    print(p)

# end if