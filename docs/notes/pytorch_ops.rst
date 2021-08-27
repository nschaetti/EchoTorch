Compatibility between TimeTensors and PyTorch operations
========================================================

This page list the compatibility with the different PyTorch operations on tensor.

.. contents:: Table of Contents

.. _Summary:

Summary
~~~~~~~

==========================  =======================================  =======================================================
PyTorch Ops                 Inputs                                   Outputs
==========================  =======================================  =======================================================
:ref:`cat`                  :class:`Tensor` + :class:`TimeTensor`    If one of the ``input`` is a :class:`TimeTensor`, returns a :class:`TimeTensor`. If one of the input is a :class:`TimeTensor`, returns a :class:`TimeTensor`.
:ref:`chunk`                :class:`TimeTensor`                      Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`dsplit`               :class:`TimeTensor`                      Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`column_stack`         :class:`Tensor` + :class:`TimeTensor`    Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor` in ``intput``.
:ref:`dstack`               :class:`Tensor` + :class:`TimeTensor`    The index of the time dimension of a 0-D timeseries will increase from 0 to 1, otherwise it will stay the same.
:ref:`gather`               :class:`TimeTensor`                      Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`hsplit`               :class:`TimeTensor`                      Output :class:`TimeTensor` (s) will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`hstack`               :class:`Tensor` + :class:`TimeTensor`    Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor` in ``intput``.
:ref:`index_select`         :class:`TimeTensor`                      Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`masked_select`        :class:`TimeTensor`                      This operation will destroy the time dimension, the output will then be a :class:`Tensor`.
:ref:`movedim`              :class:`TimeTensor`                      The returned :class:`TimeTensor` will have its time dimension moved ``source`` or ``destination`` is equal to the index of the time dimension.
:ref:`moveaxis`             :class:`TimeTensor`                      The returned :class:`TimeTensor` will have its time dimension moved ``source`` or ``destination`` is equal to the index of the time dimension.
:ref:`narrow`               True                                     True
:ref:`nonzero`              True                                     True
:ref:`reshape`              :class:`TimeTensor`                      This operation will destroy the time dimension, the output will then be a :class:`Tensor`.
:ref:`row_stack`            :class:`Tensor` + :class:`TimeTensor`    True
:ref:`scatter`              True                                     True
:ref:`scatter_add`          True                                     True
:ref:`split`                True                                     True
:ref:`squeeze`              True                                     True
:ref:`stack`                True                                     Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor` in ``intput``.
:ref:`swapaxes`             True                                     See :func:`torch.tranpose`
:ref:`swapdims`             True                                     See :func:`torch.tranpose`
:ref:`t`                    True                                     True
:ref:`take`                 :class:`torch.Tensor`                    This operation will destroy the time dimension, the output will then be a :class:`Tensor`.
:ref:`take_along_dim`       :class:`torch.Tensor`                    This operation will destroy the time dimension, the output will then be a :class:`Tensor`.
:ref:`tensor_split`         True                                     True
:ref:`tile`                 True                                     True
:ref:`transpose`            True                                     True
:ref:`unbind`               True                                     Output :class:`TimeTensor` (s) will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`unsqueeze`            True                                     True
:ref:`vsplit`               True                                     True
:ref:`vstack`               :class:`Tensor` + :class:`TimeTensor`    True
:ref:`where`                True                                     True
==========================  =======================================  =======================================================

.. _Indexing, Slicing, Joining, Mutating Ops:

Indexing, Slicing, Joining, Mutating Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _cat:

cat
^^^

:func:`torch.cat` concatenates a given sequence of `seq` tensors in the given dimension. With :class:`TimeTensor`, the
time dimension will be kept as the number of dimension of each object in the ``input`` must be the same. If multiple
timetensors are given in the input with different time dimension, only the one of the first timetensor is taken into
account. Example:

    >>> ...

If you want to concatenate timetensors directly on the time dimension, check :func:`echotorch.tcat()`.

.. _chunk:

chunk
^^^^^

.. _dsplit:

dsplit
^^^^^^

.. _column_stack:

column_stack
^^^^^^^^^^^^

:func:`torch.column_stack()` can take a list :class:`torch.Tensor` and :class:`TimeTensor` and stack them horizontally,
meaning on the second dimension (``dim=1``). 1-D :class:`torch.Tensor` will be reshaped to ``(t.numel(), 1)`` and
0-D timeseries to 1-D before stacking.

For example, if you create a :class:`torch.Tensor` of shape :math:`(T, 2)` and a :class:`TimeTensor` for a 0D timeseries
of length :math:`T`, the output of :func:`torch.column_stack()` will be a :class:`TimeTensor` with the time dimension
at the same index and same length as the first :class:`TimeTensor` in the ``input``. Let's create a first
:class:`torch.Tensor` of shape :math:`(10, 2)`.

    >>> x = torch.arange(20).reshape(10, 2)

Now, let's create a :class:`TimeTensor` for a 0-D timeseries of length 10.

    >>> z = echotorch.arange(10)

We then use :func:`torch.column_stack()` to stack them on the second dimension, which is here a channel dimension. The
output will be a :class:`TimeTensor` of length 10 with one channel dimension of size 3, the first :class:`TimeTensor`
being transformed into a 1-D timeseries by the operation.

    >>> torch.column_stack((z, x))
    timetensor([[ 0,  0,  1],
                [ 1,  2,  3],
                ...
                [18, 36, 37],
                [19, 38, 39]], time_dim: 0)

However, :func:`torch.column_stack()` applied to :class:`TimeTensor` does not stack ``input`` on the **time dimension**
but on the second dimension. If the time dimension is at position 1 (``dim=1``), this operation will stack the
``input`` on the time dimension, if the second dimension is a **batch dimension**, this operation will stack on this
batch dimension.

To stack :class:`torch.Tensor` and :class:`TimeTensor` on the time dimension, see :func:`echotorch.tstack()`.

.. _dstack:

dstack
^^^^^^

.. _slicing_indexing:


``slicing_indexing``
~~~~~~~~~~~~~~~~~~~~

