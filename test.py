
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import echotorch.nn


if __name__ == "__main__":

    reservoir_size = 100
    input_size = 2

    x = Variable(torch.rand(reservoir_size, reservoir_size), requires_grad=False)
    u = Variable(torch.ones(input_size), requires_grad=True)
    win = Variable(torch.rand(reservoir_size, input_size), requires_grad=False)
    w = Variable(torch.rand(reservoir_size, reservoir_size), requires_grad=False)
    winu = win.mv(u)
    print("winu : ")
    print(winu)
    wx = w.mm(x)
    print("wx : ")
    print(wx)
    x = F.tanh(winu + wx)
    print(x)
    print(x.creator)

# end if