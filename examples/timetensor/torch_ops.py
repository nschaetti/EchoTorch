

import torch
import echotorch


def print_var(head, t_in):
    if isinstance(t_in, echotorch.TimeTensor):
        print("{}: {}, {}, time_dim: {}, tlen: {}, csize: {}, bsize: {}".format(head, t_in.size(), t_in.__class__.__name__, t_in.time_dim, t_in.tlen, t_in.csize(), t_in.bsize()))
    elif isinstance(t_in, torch.Tensor):
        print("{}: {}, {}".format(head, t_in.size(), t_in.__class__.__name__))
    elif isinstance(t_in, list) or isinstance(t_in, tuple):
        for el_i, el in enumerate(t_in):
            print_var("{}:{}".format(head, el_i), el)
        # end for
    # end if
# end print_var


# Create tensor and time tensors
ptch = torch.randn(100, 2)
echt = echotorch.randn(2, length=100)

# Sizes
# print("ptch.size(): {}".format(ptch.size()))
# print("echt.size(): {}".format(echt.size()))

# # Is tensor
# print("is_tensor(ptch): {}".format(torch.is_tensor(ptch)))
# print("is_tensor(echt): {}".format(torch.is_tensor(echt)))

# # Numel
# print("numel(ptch): {}".format(torch.numel(ptch)))
# print("numel(echt): {}".format(torch.numel(echt)))

# As tensor (doesn't work)
# print("as_tensor(ptch): {}".format(torch.as_tensor(ptch)))
# print("as_tensor(echt): {}".format(torch.as_tensor(echt)))

# Cat
print("=======================")
print("cat")
x = torch.randn(3, 2)
y = echotorch.randn(2, length=3)
z = echotorch.as_timetensor(torch.randn(10, 2), time_dim=1)
print_var("x", x)
print_var("y", y)
print_var("z", z)
# print("in x: {}, {}".format(x.size(), type(x)))
# print("in y: {}, {}, time_dim: {}, tlen: {}".format(y.size(), type(y), y.time_dim, y.tlen))
out = torch.cat((x, x, x), 0)
print_var("out", out)
out = torch.cat((x, x, x), 1)
print_var("out", out)
out = torch.cat((x, y, x), 0)
print_var("out", out)
out = torch.cat((x, y, x), 1)
print_var("out", out)
# Must raise a RuntimeError
# out = torch.cat((y, z), 0)
# print_var("out", out)
print("")

# chunk
print("=======================")
print("chunk")
x = torch.randn(10, 2)
y = echotorch.randn(2, length=10)
z = echotorch.as_timetensor(torch.randn(10, 2), time_dim=1)
print_var("x", x)
print_var("y", y)
print_var("z", z)
out = torch.chunk(x, 3, 0)
print_var("out", out)
out = torch.chunk(y, 3, 0)
print_var("out", out)
print("")

# dsplit
print("=======================")
print("dsplit")
x = torch.arange(16.0).reshape(2, 2, 4)
y = echotorch.timetensor(x, time_dim=2)
z = echotorch.timetensor(x, time_dim=1)
print_var("x", x)
out = torch.dsplit(x, 2)
print_var("out", out)
out = torch.dsplit(x, [1, 3])
print_var("out", out)
out = torch.dsplit(y, [1, 3])
print_var("out", out)
out = torch.dsplit(z, [1, 3])
print_var("out", out)
print("")

# column_stack
print("=======================")
print("column_stack")
x = torch.arange(20)
y = torch.arange(40).reshape(20, 2)
z = echotorch.arange(20)
print_var("x", x)
print_var("y", y)
print_var("z", z)
out = torch.column_stack((x, y, y))
print_var("out", out)
out = torch.column_stack((z, y))
print_var("out", out)
print(out)
print("")

# dstack
print("=======================")
print("dstack")
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
z = echotorch.timetensor([7, 8, 9])
print_var("x", x)
print_var("y", y)
print_var("z", z)
out = torch.dstack((x, z))
print_var("out", out)
out = torch.dstack((x, y))
print_var("out", out)
x = torch.tensor([[1], [2], [3]])
y = torch.tensor([[4], [5], [6]])
print_var("x", x)
print_var("y", y)
out = torch.dstack((x, y))
print_var("out", out)
x = torch.tensor([[1, 2, 3]])
y = torch.tensor([[4, 5, 6]])
print_var("x", x)
print_var("y", y)
out = torch.dstack((x, y))
print_var("out", out)
x = torch.tensor([[[1, 2, 3]]])
y = torch.tensor([[[4, 5, 6]]])
print_var("x", x)
print_var("y", y)
out = torch.dstack((x, y))
print_var("out", out)
x = torch.tensor([[1], [2], [3]])
z = echotorch.timetensor([[7], [8], [9]])
print_var("x", x)
print_var("z", z)
out = torch.dstack((x, z))
print_var("out", out)
print("")

# gather
print("=======================")
print("gather")
x = torch.tensor([[1, 2], [3, 4]])
print_var("x", x)
out = torch.gather(x, 1, torch.tensor([[0, 0], [1, 0]]))
print_var("out", out)
z = echotorch.timetensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [12, 13]])
print_var("z", z)
out = torch.gather(z, 1, torch.tensor([[0, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]))
print_var("out", out)

print("")


# at least 3d
print("=======================")
print("at_leat_3d")
x = torch.randn(2)
print_var("x", x)
out = torch.atleast_3d(x)
print_var("out", out)
x = torch.randn(2, 2)
print_var("x", x)
out = torch.atleast_3d(x)
print_var("out", out)
x = torch.randn(2, 2, 2)
print_var("x", x)
out = torch.atleast_3d(x)
print_var("out", out)
x = torch.tensor([])
print_var("x", x)
out = torch.atleast_3d(x)
print_var("out", out)
z = echotorch.randn(length=2)
print_var("z", z)
out = torch.atleast_3d(z)
print_var("out", out)
z = echotorch.randn(1, length=2)
print_var("z", z)
out = torch.atleast_3d(z)
print_var("out", out)
print("")

# hsplit
print("=======================")
print("hsplit")
x = torch.arange(16.0).reshape(4, 4)
z = echotorch.timetensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
print_var("x", x)
print_var("z", z)
out = torch.hsplit(x, 2)
print_var("out", out)
out = torch.hsplit(z, 2)
print_var("out", out)
x = torch.arange(16.0)
print_var("x", x)
out = torch.hsplit(x, 8)
print_var("out", out)
z = echotorch.arange(16.0)
print_var("z", z)
out = torch.hsplit(z, 8)
print_var("out", out)
print("")

# hstack
print("======================")
print("hstack")
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
z = echotorch.timetensor([7, 8, 9])
print_var("x", x)
print_var("y", y)
out = torch.hstack((x, y))
print_var("out", out)
x = torch.tensor([[1], [2], [3]])
y = torch.tensor([[4], [5], [6]])
print_var("x", x)
print_var("y", y)
out = torch.hstack((x, y))
print_var("out", out)
x = torch.tensor([1, 2, 3])
print_var("x", x)
print_var("z", z)
out = torch.hstack((x, z))
print_var("out", out)
x = torch.tensor([[1],[2],[3]])
z = echotorch.timetensor([[4],[5],[6]])
print_var("x", x)
print_var("z", z)
out = torch.hstack((x, z))
print_var("out", out)
print("")

# index_select
print("======================")
print("index_select")
x = torch.randn(3, 4)
indices = torch.tensor([0, 2])
print_var("x", x)
print(x)
print_var("indices", indices)
print(indices)
out = torch.index_select(x, 0, indices)
print_var("out", out)
print(out)
indices = torch.tensor([0])
print_var("x", x)
print(x)
print_var("indices", indices)
print(indices)
out = torch.index_select(x, 0, indices)
print_var("out", out)
print(out)
z = echotorch.randn(4, length=10)
indices = torch.tensor([0, 2, 4, 6, 8])
print_var("z", z)
print(z)
print_var("indices", indices)
print(indices)
out = torch.index_select(z, 0, indices)
print_var("out", out)
print(out)
z = echotorch.randn(4, length=10)
indices = torch.tensor([0, 2, 1, 0, 3])
print_var("z", z)
print(z)
print_var("indices", indices)
print(indices)
out = torch.index_select(z, 1, indices)
print_var("out", out)
print(out)
print("")

# masked_select
# time_dim destroyed !!
print("======================")
print("masked_select")
x = torch.randn(3, 4)
mask = x.ge(-10)
print_var("x", x)
print(x)
print_var("mask", mask)
print(mask)
out = torch.masked_select(x, mask)
print_var("out", out)
print(out)
print("")

# movedim
print("======================")
print("movedim")
x = torch.randn(3, 2, 1)
print_var("x", x)
out = torch.movedim(x, 1, 0)
print_var("out", out)
z = echotorch.randn(2, 1, length=10)
print_var("z", z)
out = torch.movedim(z, 1, 0)
print_var("out", out)
print("")

# moveaxis
print("======================")
print("moveaxis")
x = torch.randn(3, 2, 1)
print_var("x", x)
out = torch.movedim(x, 1, 0)
print_var("out", out)
z = echotorch.randn(2, 1, length=10)
print_var("z", z)
out = torch.movedim(z, 1, 0)
print_var("out", out)
print("")

# narrow
print("======================")
print("narrow")
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
z = echotorch.timetensor([[1, 2], [3, 4], [5, 6], [7, 8]])
print_var("x", x)
print_var("z", z)
print(x)
print(z)
out = torch.narrow(x, 0, 0, 2)
print_var("out", out)
print(out)
out = torch.narrow(x, 1, 2, 0)
print_var("out", out)
print(out)
out = torch.narrow(z, 0, 0, 2)
print_var("out", out)
print(out)
out = torch.narrow(z, 1, 0, 0)
print_var("out", out)
print(out)
print("")

# nonzero
# time dim destroyed !
print("=======================")
print("nonzero")
x = torch.tensor([1, 1, 1, 0, 1])
print_var("x", x)
out = torch.nonzero(x)
print_var("out", out)
print("")

# reshape
# time dim destroyed !
print("=======================")
print("reshape")
x = torch.arange(4.)
print_var("x", x)
out = torch.reshape(x, (2, 2))
print_var("out", out)
print("")

# row_stack
# alias of torch.vstack()

# scatter
print("=======================")
print("scatter")
x = torch.arange(1, 11).reshape((2, 5))
y = torch.zeros(3, 5, dtype=x.dtype)
z = echotorch.timetensor(y, time_dim=0)
print_var("x", x)
print_var("y", y)
print_var("z", z)
index = torch.tensor([[0, 1, 2, 0]])
print_var("index", index)
out = torch.scatter(y, 0, index, x)
print_var("out", out)
print(out)
out = torch.scatter(z, 0, index, x)
print_var("out", out)
print(out)
print("")

# scatter_add
print("=======================")
print("scatter_add")
x = torch.ones((2, 5))
y = torch.zeros(3, 5, dtype=x.dtype)
z = echotorch.zeros(5, length=3)
print_var("x", x)
print_var("y", y)
print_var("z", z)
index = torch.tensor([[0, 1, 2, 0, 0]])
print_var("index", index)
out = torch.scatter_add(y, 0, index, x)
print_var("out", out)
print(out)
out = torch.scatter_add(z, 0, index, x)
print_var("out", out)
print(out)
print("")

# split
print("=======================")
print("split")
x = torch.arange(10).reshape(5, 2)
z = echotorch.timetensor(x)
print_var("x", x)
print_var("z", z)
out = torch.split(x, 2)
print_var("out", out)
out = torch.split(x, [1, 4])
print_var("out", out)
out = torch.split(z, 2)
print_var("out", out)
out = torch.split(z, [1, 4])
print_var("out", out)
out = torch.split(z, 1, 1)
print_var("out", out)
print("")

# squeeze
print("=======================")
print("squeeze")
x = torch.zeros(2, 1, 2, 1, 2)
print_var("x", x)
out = torch.squeeze(x)
print_var("out", out)
z = echotorch.zeros(1, 2, 1, 2, length=2)
print_var("z", z)
out = torch.squeeze(z)
print_var("out", out)
z = echotorch.zeros(1, 2, 1, 2, length=2)
print_var("z", z)
out = torch.squeeze(z, dim=0)
print_var("out", out)
z = echotorch.zeros(1, 2, 1, 2, length=1)
print_var("z", z)
out = torch.squeeze(z)
print_var("out", out)
z = echotorch.zeros(1, 2, 1, 2, length=1)
print_var("z", z)
out = torch.squeeze(z)
print_var("out", out)
z = echotorch.zeros(1, 2, 1, 2, length=1)
print_var("z", z)
out = torch.squeeze(z, 3)
print_var("out", out)
print("")

# stack
print("=======================")
print("stack")
x = torch.zeros(100, 2)
y = torch.zeros(100, 2)
print_var("x", x)
print_var("y", y)
out = torch.stack((x, y))
print_var("out", out)
z = echotorch.zeros(2, length=100)
print_var("z", z)
out = torch.stack((x, z))
print_var("out", out)
out = torch.stack((x, z), 1)
print_var("out", out)
out = torch.stack((x, z), 2)
print_var("out", out)
z2 = echotorch.zeros(2, length=100)
z2.time_dim = 1
print_var("z2", z2)
out = torch.stack((z, z2))
print_var("out", out)
out = torch.stack((z, z2), 1)
print_var("out", out)
out = torch.stack((z, z2), 2)
print_var("out", out)
print("")

# swapaxes
# alias to transpose()

# swapdims
# alias to transpose()

# t
print("=======================")
print("t")
x = torch.randn(())
print_var("x", x)
out = torch.t(x)
print_var("out", out)
x = torch.randn(3)
print_var("x", x)
out = torch.t(x)
print_var("out", out)
x = torch.randn(2, 3)
print_var("x", x)
out = torch.t(x)
print_var("out", out)
z = echotorch.randn(length=0)
print_var("z", z)
out = torch.t(z)
print_var("out", out)
z = echotorch.randn(length=3)
print_var("z", z)
out = torch.t(z)
print_var("out", out)
z = echotorch.randn(3, length=2)
print_var("z", z)
out = torch.t(z)
print_var("out", out)
print("")

# take
# time dim destroyed!
print("=======================")
print("take")
x = torch.tensor([[4, 3, 5], [6, 7, 8]])
y = torch.tensor([[0, 2, 5], [1, 3, 4]])
print_var("x", x)
print(x)
print_var("y", y)
print(y)
out = torch.take(x, y)
print_var("out", out)
print(out)
z = echotorch.timetensor([[4, 3, 5], [6, 7, 8]], time_dim=0)
print_var("z", z)
out = torch.take(z, y)
print_var("out", out)
print(out)
print("")

# take_along_dim
# time dim destroyed!
print("=======================")
print("take_along_dim")
x = torch.tensor([[10, 30, 20], [60, 40, 50]])
print_var("x", x)
max_idx = torch.argmax(x)
print_var("max_idx", max_idx)
print(max_idx)
out = torch.take_along_dim(x, max_idx)
print_var("out", out)
print(out)
z = echotorch.timetensor([[10, 30, 20], [60, 40, 50]], time_dim=0)
print_var("z", z)
max_idx_z = torch.argmax(z)
print_var("max_idx_z", max_idx_z)
print(max_idx_z)
out = torch.take_along_dim(z, max_idx_z)
print_var("out", out)
print("")

# tensor_split
print("=======================")
print("tensor_split")
x = torch.arange(8)
print_var("x", x)
print(x)
out = torch.tensor_split(x, 3)
print_var("out", out)
print(out)
z = echotorch.arange(8)
print_var("z", z)
print(z)
out = torch.tensor_split(z, 10)
print_var("out", out)
print(out)
print("")

# tile
print("=======================")
print("tile")
x = torch.tensor([[1, 2], [3, 4]])
print_var("x", x)
print(x)
out = torch.tile(x, (2, 2))
print_var("out", out)
print(out)
z = echotorch.timetensor([[1, 2], [3, 4]], time_dim=0)
print_var("z", z)
print(z)
out = torch.tile(z, (2, 2))
print_var("out", out)
print(out)
print("")

# transpose
print("=======================")
print("transpose")
x = torch.randn(2, 3)
print_var("x", x)
print(x)
out = torch.transpose(x, 0, 1)
print_var("out", out)
print(out)
x = echotorch.randn(3, length=2)
print_var("x", x)
print(x)
out = torch.transpose(x, 0, 1)
print_var("out", out)
print(out)

# unbind
print("=======================")
print("unbind")
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print_var("x", x)
print(x)
out = torch.unbind(x)
print_var("out", out)
print(out)
x = echotorch.timetensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], time_dim=0)
print_var("x", x)
print(x)
out = torch.unbind(x)
print_var("out", out)
print(out)
x = echotorch.timetensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], time_dim=0)
print_var("x", x)
print(x)
out = torch.unbind(x, dim=1)
print_var("out", out)
print(out)
print("")

# unsqueeze
print("=========================")
print("unsqueeze")
x = torch.tensor([1, 2, 3, 4])
print_var("x", x)
print(x)
out = torch.unsqueeze(x, 0)
print_var("out", out)
print(out)
z = echotorch.timetensor([1, 2, 3, 4])
print_var("z", z)
print(z)
out = torch.unsqueeze(z, 0)
print_var("out", out)
print(out)
z = echotorch.timetensor([1, 2, 3, 4])
print_var("z", z)
print(z)
out = torch.unsqueeze(z, 1)
print_var("out", out)
print(out)
print("")

# vsplit
print("=========================")
print("vsplit")
x = torch.arange(16.0).reshape(4, 4)
print_var("x", x)
print(x)
out = torch.vsplit(x, 4)
print_var("out", out)
print(out)
z = echotorch.timetensor(torch.arange(16.0).reshape(4, 4), time_dim=0)
print_var("z", z)
print(z)
out = torch.vsplit(z, 4)
print_var("out", out)
print(out)
print("")

# vstack
print("=========================")
print("vstack")
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
print_var("x", x)
print_var("y", y)
out = torch.vstack((x, y))
print_var("out", out)
print(out)
z1 = echotorch.timetensor([1, 2, 3])
z2 = echotorch.timetensor([4, 5, 6])
print_var("z1", z1)
print_var("z2", z2)
out = torch.vstack((z1, z2))
print_var("out", out)
print(out)
z1 = echotorch.timetensor([[1], [2], [3]])
z2 = echotorch.timetensor([[4], [5], [6]])
print_var("z1", z1)
print_var("z2", z2)
out = torch.vstack((z1, z2))
print_var("out", out)
print(out)
print("")

# where
print("=========================")
print("where")
x = torch.randn(3, 2)
y = torch.ones(3, 2)
print_var("x", x)
print(x)
print_var("y", y)
print(y)
out = torch.where(x > 0, x, y)
print(x > 0)
print_var("out", out)
print(out)
z1 = echotorch.randn(2, length=10)
z2 = echotorch.ones(2, length=10)
print_var("z1", z1)
print(z1)
print_var("z2", z2)
print(z2)
out = torch.where(z1 > 0, z1, z2)
print(z1 > 0)
print_var("out", out)
print(out)
