.. currentmodule:: echotorch.base_tensors

.. _base-tensor-doc:

echotorch.base_tensors.BaseTensor
=================================

BaseTensor class reference
--------------------------

:class:`BaseTensor` is the base class of :class:`echotorch.TimeTensor`.  It wraps
a raw PyTorch :class:`torch.Tensor` and forwards most operations to it.  The
arithmetic and comparison operators return **new** tensors and leave the
original object unchanged.

.. autoclass:: BaseTensor
    :members:
    :show-inheritance:
    :special-members: __init__, __getitem__, __setitem__, __repr__,
        __eq__, __ne__, __iadd__, __isub__, __add__, __radd__,
        __sub__, __rsub__, __mul__, __rmul__, __truediv__,
        __lt__, __le__, __gt__, __ge__

Properties
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    BaseTensor.tensor

Casting
-------

The in-place casting methods cast the wrapped tensor to the requested dtype and
return ``self``:

.. autosummary::
    :toctree: generated
    :nosignatures:

    BaseTensor.double
    BaseTensor.float
    BaseTensor.half
    BaseTensor.bfloat16
    BaseTensor.bool
    BaseTensor.byte
    BaseTensor.char
    BaseTensor.short
    BaseTensor.int
    BaseTensor.long
    BaseTensor.to

Device movement and layout
--------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    BaseTensor.cuda
    BaseTensor.cpu
    BaseTensor.t

Torch function dispatch
-----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    BaseTensor.__torch_function__
    BaseTensor.__getattr__

Arithmetic and comparison operators
-----------------------------------

The arithmetic operators return a new :class:`BaseTensor` of the same subclass
(and, for :class:`echotorch.TimeTensor`, with the same time dimension).  The
original tensor is never modified.

.. autosummary::
    :toctree: generated
    :nosignatures:

    BaseTensor.__iadd__
    BaseTensor.__isub__
    BaseTensor.__add__
    BaseTensor.__radd__
    BaseTensor.__sub__
    BaseTensor.__rsub__
    BaseTensor.__mul__
    BaseTensor.__rmul__
    BaseTensor.__truediv__
    BaseTensor.__lt__
    BaseTensor.__le__
    BaseTensor.__gt__
    BaseTensor.__ge__
    BaseTensor.__eq__
    BaseTensor.__ne__

Variants
--------

EchoTorch provides one subclass of :class:`BaseTensor` per dtype, each casting
its data on construction:

.. autosummary::
    :toctree: generated
    :nosignatures:

    DoubleBaseTensor
    FloatBaseTensor
    HalfBaseTensor
    BFloat16Tensor
    BoolBaseTensor
    ByteBaseTensor
    CharBaseTensor
    ShortBaseTensor
    IntBaseTensor
    LongBaseTensor

Constructors
------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    BaseTensor.new_basetensor
    BaseTensor.new_basetensor_with_func
