
import torch


class MyTensor(object):

    def __init__(self):
        self.attr1 = 'my_attr1'
        self._tensor = torch.ones(100)
    # end __init__

    @property
    def tensor(self):
        return self._tensor
    # end tensor

    def __getattr__(self, item):
        print("__getattr__: {}".format(item))
        if hasattr(self._tensor, item):
            return getattr(self._tensor, item)
        else:
            raise AttributeError(
                "AttributeError: Neither '{}' object nor its wrapped "
                "tensor has no attribute '{}'".format(self.__class__.__name__, item)
            )
        # end if
    # end __getattr__

    # def __getattribute__(self, item):
    #     print("__getattribute__: {}".format(item))
    # # end __getattribute__

    # Set attributes
    # def __setattr__(self, key, value):
    #     print("__setattr__: {} {}".format(key, value))
    # # end __setattr__

# end MyTensor


test = MyTensor()
# print("test: {}".format(test.test))
# print("")

print("Set requires_grad")
print(test.requires_grad)
print("1")
test.requires_grad = True
print("2")
print(test.requires_grad)
print("")

print("Set attr1")
#print(test.attr1)
print("1")
test.attr1 = "attr2"
# print("2")
print(test.attr1)

print("Call is_complex")
print(test.is_complex)=

print("attr1: {}".format(test.attr1))
print("")

print("size: {}".format(test.size()))
print("")

print("ndim: {}".format(test.ndim))
print("")

# print("other: {}".format(test.other))
# print("")

