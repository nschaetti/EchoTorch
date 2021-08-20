

import torch
import echotorch


# Create tensor and time tensors
ptch = torch.randn(100, 2)
echt = echotorch.randn(2, time_length=100)

# Sizes
print("ptch.size(): {}".format(ptch.size()))
print("echt.size(): {}".format(echt.size()))

# Is tensor
print("is_tensor(ptch): {}".format(torch.is_tensor(ptch)))
print("is_tensor(echt): {}".format(torch.is_tensor(echt)))

# Numel
print("numel(ptch): {}".format(torch.numel(ptch)))
print("numel(echt): {}".format(torch.numel(echt)))

# As tensor (doesn't work)
# print("as_tensor(ptch): {}".format(torch.as_tensor(ptch)))
# print("as_tensor(echt): {}".format(torch.as_tensor(echt)))
