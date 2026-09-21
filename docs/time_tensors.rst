Time Tensors
============

EchoTorch is built around the :class:`echotorch.TimeTensor`, a thin wrapper
around a PyTorch :class:`torch.Tensor` that remembers **which dimension is the
time dimension**.  This guide explains how to create timetensors, how the time
dimension is propagated through operations, and how the mathematical /
statistical operations defined on timetensors behave.

Creating time tensors
~~~~~~~~~~~~~~~~~~~~~

A :class:`TimeTensor` is created from a :class:`torch.Tensor` by indicating the
position of the time dimension with the ``time_dim`` argument (default ``0``)::

    >>> import torch
    >>> import echotorch

    >>> data = torch.tensor([[1., 2., 3.],
    ...                      [4., 5., 6.],
    ...                      [7., 8., 9.]])
    >>> tt = echotorch.TimeTensor(data, time_dim=0)

The helper :func:`echotorch.timetensor` (and the creation ops :func:`echotorch.rand`,
:func:`echotorch.randn`, :func:`echotorch.zeros`, :func:`echotorch.arange`, ...) build
timetensors directly::

    >>> x = echotorch.randn(5, length=20)   # 5 channels, 20 time steps

The underlying raw tensor is accessible through the :attr:`TimeTensor.tensor`
property.

Time dimension
~~~~~~~~~~~~~~

The time dimension is stored in the :attr:`TimeTensor.time_dim` attribute.  Its
size is given by :attr:`TimeTensor.tlen`.  The number of remaining *channel*
dimensions is :attr:`TimeTensor.cdim` and the number of *batch* dimensions
(ahead of the time dimension) is :attr:`TimeTensor.bdim`.

::

    >>> x = echotorch.randn(4, 8, length=20)
    >>> x.time_dim
    0
    >>> x.tlen
    20
    >>> x.cdim      # the 8-channel "feature" dimension
    1
    >>> x.bdim
    0

The time dimension can be moved afterwards with the 
:attr:`TimeTensor.time_dim` setter, provided the target index is valid.

Time dimension propagation semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EchoTorch intercepts torch operations on timetensors through
:func:`TimeTensor.__torch_function__`.  In most cases the operation is applied
to the wrapped tensor and the result is re-wrapped into a ``TimeTensor`` while
tracking how the time dimension moves:

* *Pointwise* operations (``+``, ``*``, ``torch.exp``, ...) keep the time
  dimension unchanged.
* *Dimension-changing* operations (``squeeze``, ``unsqueeze``, ``permute``,
  ``transpose``, ``flatten``, ``reshape``, ``stack``, ``t``, ...) update
  ``time_dim`` accordingly through the ``after_*`` callbacks.
* *Reduction* operations (``torch.mean``, ``torch.sum``, ``torch.std``, ...)
  return a plain :class:`torch.Tensor` when they remove the time dimension, and
  a :class:`TimeTensor` when the time dimension is preserved (e.g. with
  ``keepdim=True`` or when reducing another dimension).

This is handled automatically; when calling these operations you do not need to
track the time dimension yourself.

Manipulating dimensions
~~~~~~~~~~~~~~~~~~~~~~~~

Dimension manipulation is provided both by the corresponding torch functions
on timetensors (``torch.stack``, ``torch.permute``, ``torch.transpose``,
``torch.flatten``, ``torch.reshape``, ...) and by a set of dedicated EchoTorch
helpers (:func:`echotorch.tcat`, :func:`echotorch.cat`,
:func:`echotorch.tindex_select`, :func:`echotorch.as_strided`), documented in
the :doc:`module reference <echotorch>`.

Statistical operations
~~~~~~~~~~~~~~~~~~~~~~~

EchoTorch provides time-aware statistical operations that reduce over the time
dimension (see the :doc:`module reference <echotorch>` for the complete
list):

    >>> x = echotorch.randn(5, length=100)
    >>> echotorch.tmean(x)      # mean of each channel over time
    tensor([...])
    >>> echotorch.tstd(x)       # standard deviation (Bessel's correction by default)
    tensor([...])
    >>> echotorch.cov(x)        # covariance matrix of the 5 channels
    tensor([[...]])

Auto-covariance and (cross-)correlation coefficients are available in
:mod:`echotorch.acf` (see :doc:`echotorch.acf`).
