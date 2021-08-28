

# Imports
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


# atleast_1d
# return timetensor, same data, same time dim
print("-------------------------")
print("atleast_1d")
x = torch.tensor(1.)
print_var("x", x)
print(x)
out = torch.atleast_1d(x)
print_var("out", out)
print(out)
z = echotorch.timetensor([1, 2, 3])
print_var("z", z)
print(z)
out = torch.atleast_1d(z)
print_var("out", out)
print(out)
z = echotorch.timetensor([[1], [2], [3]])
print_var("z", z)
print(z)
out = torch.atleast_1d(z)
print_var("out", out)
print(out)
print("")

# atleast_2d
# when 0-D timeserie, add a batch dim and time_dim + 1
print("-------------------------")
print("atleast_2d")
x = torch.tensor(1.)
print_var("x", x)
print(x)
out = torch.atleast_2d(x)
print_var("out", out)
print(out)
z = echotorch.timetensor([1, 2, 3])
print_var("z", z)
print(z)
out = torch.atleast_2d(z)
print_var("out", out)
print(out)
z = echotorch.timetensor([[1], [2], [3]])
print_var("z", z)
print(z)
out = torch.atleast_2d(z)
print_var("out", out)
print(out)
print("")

# atleast_3d
# when 0-D timeserie, add a batch dim and time_dim + 1 and a channel dim
# when 1-D timseries, add a channel dim
print("-------------------------")
print("atleast_3d")
x = torch.tensor(1.)
print_var("x", x)
print(x)
out = torch.atleast_3d(x)
print_var("out", out)
print(out)
z = echotorch.timetensor([1, 2, 3])
print_var("z", z)
print(z)
out = torch.atleast_3d(z)
print_var("out", out)
print(out)
z = echotorch.timetensor([[1], [2], [3]])
print_var("z", z)
print(z)
out = torch.atleast_3d(z)
print_var("out", out)
print(out)
print("")
z = echotorch.timetensor([[[1]], [[2]], [[3]]])
print_var("z", z)
print(z)
out = torch.atleast_3d(z)
print_var("out", out)
print(out)
print("")

# bincount
# destroy time dim
print("-------------------------")
print("bincount")
x = torch.randint(0, 8, (5,), dtype=torch.int64)
y = torch.linspace(0, 1, steps=5)
print(x), print(y)
torch.bincount(x)
x.bincount(y)
z = echotorch.timetensor([1, 2, 3])
print(torch.bincount(z))
print("")

# block_diag
# keep time dim of first timetensor
print("-------------------------")
print("block_diag")
A = torch.tensor([[0, 1], [1, 0]])
B = echotorch.timetensor([[3, 4, 5], [6, 7, 8]])
C = torch.tensor(7)
D = torch.tensor([1, 2, 3])
E = torch.tensor([[4], [5], [6]])
print(torch.block_diag(A, B, C, D, E))
print("")

# broadcast_tensors
print("-------------------------")
print("broadcast_tensors")
x = torch.arange(3).view(1, 3)
print(x)
# y = torch.arange(3).view(3, 1)
y = torch.tensor([[4], [5]])
print(y)
a, b = torch.broadcast_tensors(x, y)
print(a.size())
print(a)
print(b.size())
print(b)
print("##")
x = echotorch.timetensor([[0, 1, 2]])
a, b = torch.broadcast_tensors(x, y)
print(x)
print(a.size())
print(a)
print(b.size())
print(b)
print("##")
z = echotorch.timetensor([[4], [5]], time_dim=1)
a, b = torch.broadcast_tensors(x, z)
print(x)
print(z)
print(a.size())
print(a)
print(b.size())
print(b)
print("")

# broadcast_tensors
# no input tensor
print("-------------------------")
print("broadcast_to")
x = torch.tensor([1, 2, 3])
out = torch.broadcast_to(x, (5, 3))
print(x)
print(out)
print(out.size())
z = echotorch.timetensor([1, 2, 3])
out = torch.broadcast_to(z, (5, 3))
print(z)
print(out)
print(out.size())
print("")

# bucketize
print("-------------------------")
print("bucketize")
boundaries = torch.tensor([1, 3, 5, 7, 8])
v = torch.tensor([[3, 6, 9], [3, 6, 9]])
print(torch.bucketize(v, boundaries, right=False))
z = echotorch.timetensor([[3, 6, 9], [3, 6, 9]])
print(torch.bucketize(z, boundaries, right=False))
print("")

# cartesian_prod
print("-------------------------")
print("cartesian_prod")
a = [1, 2, 3]
b = [4, 5]
c = [6, 7]
tensor_a = torch.tensor(a)
tensor_b = torch.tensor(b)
tensor_c = torch.tensor(c)
print(torch.cartesian_prod(tensor_a, tensor_b, tensor_c))
ttensor_a = echotorch.timetensor(a)
ttensor_b = echotorch.timetensor(b)
ttensor_c = echotorch.timetensor(c)
print(torch.cartesian_prod(ttensor_a, ttensor_b, ttensor_c))
print("")

# cdist
print("-------------------------")
print("cdist")
x = echotorch.timetensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
y = echotorch.timetensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])
print(torch.cdist(x, y, p=2))
print("")

# clone
print("-------------------------")
print("clone")
x = echotorch.timetensor([1, 3, 5, 7, 8])
y = torch.clone(x)
print(id(x))
print(id(y))
print(y)
print("")


