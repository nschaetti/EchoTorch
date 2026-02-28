
import torch
import echotorch

x = torch.arange(1., 6.)
print(x)
print(torch.topk(x, 3))

y = echotorch.arange(5, 0, -1)
y = torch.unsqueeze(y, dim=1)
print(y)
y = torch.tile(y, (1, 2))
print(y)
print(torch.topk(y, 1, 0))
print(torch.topk(y, 1, 1))

print(torch.msort(y))
