Compatibility between TimeTensors and PyTorch operations
========================================================

This page list the compatibility with the different PyTorch operations on tensor.

.. contents:: Table of Contents

.. _Summary:

Summary
~~~~~~~

Indexing, Slicing, Joining, Mutating Ops
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

===============================  =======================================  =======================================================
PyTorch Ops                      Inputs                                   Outputs
===============================  =======================================  =======================================================
:ref:`cat`                       :class:`Tensor` + :class:`TimeTensor`    If one of the ``input`` is a :class:`TimeTensor`, returns a :class:`TimeTensor`. If one of the input is a :class:`TimeTensor`, returns a :class:`TimeTensor`.
:ref:`chunk`                     :class:`TimeTensor`                      Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`dsplit`                    :class:`TimeTensor`                      Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`column_stack`              :class:`Tensor` + :class:`TimeTensor`    Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor` in ``intput``.
:ref:`dstack`                    :class:`Tensor` + :class:`TimeTensor`    The index of the time dimension of a 0-D timeseries will increase from 0 to 1, otherwise it will stay the same.
:ref:`gather`                    :class:`TimeTensor`                      Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`hsplit`                    :class:`TimeTensor`                      Output :class:`TimeTensor` (s) will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`hstack`                    :class:`Tensor` + :class:`TimeTensor`    Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor` in ``intput``.
:ref:`index_select`              :class:`TimeTensor`                      Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`masked_select`             :class:`TimeTensor`                      This operation will destroy the time dimension, the output will then be a :class:`Tensor`.
:ref:`movedim`                   :class:`TimeTensor`                      The returned :class:`TimeTensor` will have its time dimension moved ``source`` or ``destination`` is equal to the index of the time dimension.
:ref:`moveaxis`                  :class:`TimeTensor`                      The returned :class:`TimeTensor` will have its time dimension moved ``source`` or ``destination`` is equal to the index of the time dimension.
:ref:`narrow`                    TODO                                     TODO
:ref:`nonzero`                   TODO                                     TODO
:ref:`reshape`                   :class:`TimeTensor`                      This operation will destroy the time dimension, the output will then be a :class:`Tensor`.
:ref:`row_stack`                 :class:`Tensor` + :class:`TimeTensor`    TODO
:ref:`scatter`                   TODO                                     TODO
:ref:`scatter_add`               TODO                                     TODO
:ref:`split`                     TODO                                     TODO
:ref:`squeeze`                   TODO                                     TODO
:ref:`stack`                     TODO                                     Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor` in ``intput``.
:ref:`swapaxes`                  TODO                                     See :func:`torch.transpose`
:ref:`swapdims`                  TODO                                     See :func:`torch.transpose`
:ref:`t`                         TODO                                     TODO
:ref:`take`                      :class:`torch.Tensor`                    This operation will destroy the time dimension, the output will then be a :class:`Tensor`.
:ref:`take_along_dim`            :class:`torch.Tensor`                    This operation will destroy the time dimension, the output will then be a :class:`Tensor`.
:ref:`tensor_split`              TODO                                     TODO
:ref:`tile`                      TODO                                     TODO
:ref:`transpose`                 TODO                                     TODO
:ref:`unbind`                    TODO                                     Output :class:`TimeTensor` (s) will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`unsqueeze`                 TODO                                     TODO
:ref:`vsplit`                    TODO                                     TODO
:ref:`vstack`                    :class:`Tensor` + :class:`TimeTensor`    TODO
:ref:`where`                     TODO                                     TODO
===============================  =======================================  =======================================================

Pointwise Ops
^^^^^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Outputs
=============================================================  ===============================================================  =======================================
:ref:`abs`                                                     :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`absolute`                                                :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`acos`                                                    :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`arccos`                                                  :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`acosh`                                                   :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`arccosh`                                                 :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`add`                                                     :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`addcdiv`                                                 :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`addcmul`                                                 :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`angle`                                                   :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`asin`                                                    :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`arcsin`                                                  :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`asinh`                                                   :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`arcsinh`                                                 :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`atan`                                                    :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`arctan`                                                  :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`atanh`                                                   :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`arctanh`                                                 :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`atan2`                                                   :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`bitwise_not`                                             :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`bitwise_and`                                             :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`bitwise_or`                                              :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`bitwise_xor`                                             :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`ceil`                                                    :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`clamp`                                                   :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`clip`                                                    :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`conj`                                                    :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`copysign`                                                :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`cos`                                                     :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`cosh`                                                    :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`deg2rad`                                                 :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`div`                                                     :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`divide`                                                  :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`digamma`                                                 :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`erf`                                                     :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`erfc`                                                    :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`erfinv`                                                  :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`exp`                                                     :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`exp2`                                                    :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`expm1`                                                   :class:`TimeTensor`                                              :class:`TimeTensor`
:ref:`fake_quantize_per_channel_affine`                        TODO                                                             TODO
:ref:`fake_quantize_per_tensor_affine`                         TODO                                                             TODO
:ref:`fix`                                                     TODO                                                             TODO
:ref:`float_power`                                             TODO                                                             TODO
:ref:`floor`                                                   TODO                                                             TODO
:ref:`floor_divide`                                            TODO                                                             TODO
:ref:`fmod`                                                    TODO                                                             TODO
:ref:`frac`                                                    TODO                                                             TODO
:ref:`frexp`                                                   TODO                                                             TODO
:ref:`gradient`                                                TODO                                                             TODO
:ref:`imag`                                                    TODO                                                             TODO
:ref:`ldexp`                                                   TODO                                                             TODO
:ref:`lerp`                                                    TODO                                                             TODO
:ref:`lgamma`                                                  TODO                                                             TODO
:ref:`log`                                                     TODO                                                             TODO
:ref:`log10`                                                   TODO                                                             TODO
:ref:`log1p`                                                   TODO                                                             TODO
:ref:`log2`                                                    TODO                                                             TODO
:ref:`logaddexp`                                               TODO                                                             TODO
:ref:`logaddexp2`                                              TODO                                                             TODO
:ref:`logical_and`                                             TODO                                                             TODO
:ref:`logical_not`                                             TODO                                                             TODO
:ref:`logical_or`                                              TODO                                                             TODO
:ref:`logical_xor`                                             TODO                                                             TODO
:ref:`logit`                                                   TODO                                                             TODO
:ref:`hypot`                                                   TODO                                                             TODO
:ref:`i0`                                                      TODO                                                             TODO
:ref:`igamma`                                                  TODO                                                             TODO
:ref:`mul`                                                     TODO                                                             TODO
:ref:`multiply`                                                TODO                                                             TODO
:ref:`mvlgamma`                                                TODO                                                             TODO
:ref:`nan_to_num`                                              TODO                                                             TODO
:ref:`neg`                                                     TODO                                                             TODO
:ref:`negative`                                                TODO                                                             TODO
:ref:`nextafter`                                               TODO                                                             TODO
:ref:`polygamma`                                               TODO                                                             TODO
:ref:`positive`                                                TODO                                                             TODO
:ref:`pow`                                                     TODO                                                             TODO
:ref:`rad2deg`                                                 TODO                                                             TODO
:ref:`real`                                                    TODO                                                             TODO
:ref:`reciprocal`                                              TODO                                                             TODO
:ref:`remainder`                                               TODO                                                             TODO
:ref:`round`                                                   TODO                                                             TODO
:ref:`rsqrt`                                                   TODO                                                             TODO
:ref:`sigmoid`                                                 TODO                                                             TODO
:ref:`sign`                                                    TODO                                                             TODO
:ref:`sgn`                                                     TODO                                                             TODO
:ref:`signbit`                                                 TODO                                                             TODO
:ref:`sin`                                                     TODO                                                             TODO
:ref:`sinc`                                                    TODO                                                             TODO
:ref:`sinh`                                                    TODO                                                             TODO
:ref:`sqrt`                                                    TODO                                                             TODO
:ref:`square`                                                  TODO                                                             TODO
:ref:`sub`                                                     TODO                                                             TODO
:ref:`subtract`                                                TODO                                                             TODO
:ref:`tan`                                                     TODO                                                             TODO
:ref:`tanh`                                                    TODO                                                             TODO
:ref:`true_divide`                                             TODO                                                             TODO
:ref:`trunc`                                                   TODO                                                             TODO
:ref:`xlogy`                                                   TODO                                                             TODO
=============================================================  ===============================================================  =======================================

Reduction Ops
^^^^^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Outputs
=============================================================  ===============================================================  =======================================
:ref:`argmax`                                                  TODO                                                             TODO
:ref:`argmin`                                                  TODO                                                             TODO
:ref:`amax`                                                    TODO                                                             TODO
:ref:`amin`                                                    TODO                                                             TODO
:ref:`all`                                                     TODO                                                             TODO
:ref:`any`                                                     TODO                                                             TODO
:ref:`max`                                                     TODO                                                             TODO
:ref:`dist`                                                    TODO                                                             TODO
:ref:`logsumexp`                                               TODO                                                             TODO
:ref:`mean`                                                    TODO                                                             TODO
:ref:`median`                                                  TODO                                                             TODO
:ref:`nanmedian`                                               TODO                                                             TODO
:ref:`mode`                                                    TODO                                                             TODO
:ref:`norm`                                                    TODO                                                             TODO
:ref:`nansum`                                                  TODO                                                             TODO
:ref:`prod`                                                    TODO                                                             TODO
:ref:`quantile`                                                TODO                                                             TODO
:ref:`nanquantile`                                             TODO                                                             TODO
:ref:`std`                                                     TODO                                                             TODO
:ref:`std_mean`                                                TODO                                                             TODO
:ref:`sum`                                                     TODO                                                             TODO
:ref:`unique`                                                  TODO                                                             TODO
:ref:`unique_consecutive`                                      TODO                                                             TODO
:ref:`var`                                                     TODO                                                             TODO
:ref:`var_mean`                                                TODO                                                             TODO
:ref:`count_nonzero`                                           TODO                                                             TODO
=============================================================  ===============================================================  =======================================

Comparison Ops
^^^^^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Outputs
=============================================================  ===============================================================  =======================================
:ref:`allclose`                                                TODO                                                             TODO
:ref:`argsort`                                                 TODO                                                             TODO
:ref:`eq`                                                      TODO                                                             TODO
:ref:`equal`                                                   TODO                                                             TODO
:ref:`ge`                                                      TODO                                                             TODO
:ref:`greater_equal`                                           TODO                                                             TODO
:ref:`gt`                                                      TODO                                                             TODO
:ref:`greated`                                                 TODO                                                             TODO
:ref:`isclose`                                                 TODO                                                             TODO
:ref:`isfinite`                                                TODO                                                             TODO
:ref:`isinf`                                                   TODO                                                             TODO
:ref:`isposinf`                                                TODO                                                             TODO
:ref:`isneginf`                                                TODO                                                             TODO
:ref:`isnan`                                                   TODO                                                             TODO
:ref:`isreal`                                                  TODO                                                             TODO
:ref:`kthvalue`                                                TODO                                                             TODO
:ref:`le`                                                      TODO                                                             TODO
:ref:`less_equal`                                              TODO                                                             TODO
:ref:`lt`                                                      TODO                                                             TODO
:ref:`less`                                                    TODO                                                             TODO
:ref:`maximum`                                                 TODO                                                             TODO
:ref:`minimum`                                                 TODO                                                             TODO
:ref:`fmax`                                                    TODO                                                             TODO
:ref:`fmin`                                                    TODO                                                             TODO
:ref:`ne`                                                      TODO                                                             TODO
:ref:`not_equal`                                               TODO                                                             TODO
:ref:`sort`                                                    TODO                                                             TODO
:ref:`topk`                                                    TODO                                                             TODO
:ref:`msort`                                                   TODO                                                             TODO
=============================================================  ===============================================================  =======================================

Spectral Ops
^^^^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Outputs
=============================================================  ===============================================================  =======================================
:ref:`stft`                                                    TODO                                                             TODO
:ref:`istft`                                                   TODO                                                             TODO
:ref:`bertlett_window`                                         TODO                                                             TODO
:ref:`blackman_window`                                         TODO                                                             TODO
:ref:`hamming_window`                                          TODO                                                             TODO
:ref:`hann_window`                                             TODO                                                             TODO
:ref:`kaiser_window                                            TODO                                                             TODO
=============================================================  ===============================================================  =======================================

Other operations
^^^^^^^^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Outputs
=============================================================  ===============================================================  =======================================
:ref:`atleast_1d`                                              :class:`TimeTensor`                                              :class:`TimeTensor` are already at least 1D, this operation returns the same timetensor.
:ref:`atleast_2d`                                              :class:`TimeTensor`                                              When ``input`` is a 0-D timeseries, a batch dimension is added and the index of the time dimension is incremented by 1.
:ref:`atleast_3d`                                              :class:`TimeTensor`                                              When ``input`` is a 0-D timeseries, a batch and a channel dimension are added and the index of the time dimension is incremented by 1. When ``input`` is a 1-D timeseries, only the channel dimension is added a not increment is made to the index of the time dimension.
:ref:`bincount`                                                :class:`TimeTensor`                                              This operation destroys the time dimension, it then returns a :class:`Tensor`.
:ref:`block_diag`                                              :class:`torch.Tensor` + :class:`TimeTensor`                      Returns a :class:`TimeTensor` with the index of the time dimension of the first timetensor in the list.
:ref:`broadcast_tensors`                                       :class:`torch.Tensor` + :class:`TimeTensor`                      :class:`TimeTensor` in the ``input`` list is returned broadcasted as a :class:`TimeTensor` with same time index, :class:`Tensor` are returned broadcasted as :class:`Tensor`.
:ref:`broadcast_to`                                            :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`cartesian_prod`                                          :class:`torch.Tensor` + :class:`TimeTensor`                      Output :class:`TimeTensor` will have an time dimension index set to 0.
:ref:`clone`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have an time dimension index set to 0.
:ref:`combinations`                                            TODO                                                             TODO
:ref:`cross`                                                   TODO                                                             TODO
:ref:`cummax`                                                  TODO                                                             TODO
:ref:`cummin`                                                  TODO                                                             TODO
:ref:`cumprod`                                                 TODO                                                             TODO
:ref:`cumsum`                                                  TODO                                                             TODO
:ref:`diag`                                                    TODO                                                             TODO
:ref:`diag_embed`                                              TODO                                                             TODO
:ref:`diagflat`                                                TODO                                                             TODO
:ref:`diagonal`                                                TODO                                                             TODO
:ref:`diff`                                                    TODO                                                             TODO
:ref:`einsum`                                                  TODO                                                             TODO
:ref:`flatten`                                                 TODO                                                             TODO
:ref:`flip`                                                    TODO                                                             TODO
:ref:`fliplr`                                                  TODO                                                             TODO
:ref:`flipud`                                                  TODO                                                             TODO
:ref:`kron`                                                    TODO                                                             TODO
:ref:`rot90`                                                   TODO                                                             TODO
:ref:`gcd`                                                     TODO                                                             TODO
:ref:`histc`                                                   TODO                                                             TODO
:ref:`meshgrid`                                                TODO                                                             TODO
:ref:`lcm`                                                     TODO                                                             TODO
:ref:`logcumsumexp`                                            TODO                                                             TODO
:ref:`ravel`                                                   TODO                                                             TODO
:ref:`renorm`                                                  TODO                                                             TODO
:ref:`repeat_interleave`                                       TODO                                                             TODO
:ref:`roll`                                                    TODO                                                             TODO
:ref:`searchsorted`                                            TODO                                                             TODO
:ref:`tensordot`                                               TODO                                                             TODO
:ref:`trace`                                                   TODO                                                             TODO
:ref:`tril`                                                    TODO                                                             TODO
:ref:`tril_indices`                                            TODO                                                             TODO
:ref:`triu`                                                    TODO                                                             TODO
:ref:`triu_indices`                                            TODO                                                             TODO
:ref:`vander`                                                  TODO                                                             TODO
:ref:`view_as_real`                                            TODO                                                             TODO
:ref:`view_as_complex`                                         TODO                                                             TODO
=============================================================  ===============================================================  =======================================

BLAS and LAPACK Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Outputs
=============================================================  ===============================================================  =======================================
:ref:`addbmm`                                                  TODO                                                             TODO
:ref:`addmm`                                                   TODO                                                             TODO
:ref:`addmv`                                                   TODO                                                             TODO
:ref:`addr`                                                    TODO                                                             TODO
:ref:`baddbmm`                                                 TODO                                                             TODO
:ref:`bmm`                                                     TODO                                                             TODO
:ref:`chain_matmul`                                            TODO                                                             TODO
:ref:`cholesky`                                                TODO                                                             TODO
:ref:`cholesky_inverse`                                        TODO                                                             TODO
:ref:`cholesky_solve`                                          TODO                                                             TODO
:ref:`dot`                                                     TODO                                                             TODO
:ref:`eig`                                                     TODO                                                             TODO
:ref:`geqrf`                                                   TODO                                                             TODO
:ref:`ger`                                                     TODO                                                             TODO
:ref:`inner`                                                   TODO                                                             TODO
:ref:`inverse`                                                 TODO                                                             TODO
:ref:`det`                                                     TODO                                                             TODO
:ref:`logdet`                                                  TODO                                                             TODO
:ref:`slogdet`                                                 TODO                                                             TODO
:ref:`lstsq`                                                   TODO                                                             TODO
:ref:`lu`                                                      TODO                                                             TODO
:ref:`lu_solve`                                                TODO                                                             TODO
:ref:`lu_unpack`                                               TODO                                                             TODO
:ref:`matmul`                                                  TODO                                                             TODO
:ref:`matrix_power`                                            TODO                                                             TODO
:ref:`matrix_rank`                                             TODO                                                             TODO
:ref:`matrix_exp`                                              TODO                                                             TODO
:ref:`mm`                                                      TODO                                                             TODO
:ref:`mv`                                                      TODO                                                             TODO
:ref:`orgqr`                                                   TODO                                                             TODO
:ref:`ormqr`                                                   TODO                                                             TODO
:ref:`outer`                                                   TODO                                                             TODO
:ref:`pinverse`                                                TODO                                                             TODO
:ref:`qr`                                                      TODO                                                             TODO
:ref:`solve`                                                   TODO                                                             TODO
:ref:`svd`                                                     TODO                                                             TODO
:ref:`svd_lowrank`                                             TODO                                                             TODO
:ref:`pca_lowrank`                                             TODO                                                             TODO
:ref:`symeig`                                                  TODO                                                             TODO
:ref:`lobpcg`                                                  TODO                                                             TODO
:ref:`trapz`                                                   TODO                                                             TODO
:ref:`triangular_solve`                                        TODO                                                             TODO
:ref:`vdot`                                                    TODO                                                             TODO
=============================================================  ===============================================================  =======================================

Utilities
^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Outputs
=============================================================  ===============================================================  =======================================
:ref:`compiled_with_cxx11_abi`                                 TODO                                                             TODO
:ref:`result_type`                                             TODO                                                             TODO
:ref:`can_cast`                                                TODO                                                             TODO
:ref:`promote_types`                                           TODO                                                             TODO
:ref:`use_deterministic_algorithms`                            TODO                                                             TODO
:ref:`are_deterministic_algorithms_enabled`                    TODO                                                             TODO
:ref:`set_warn_always`                                         TODO                                                             TODO
:ref:`is_warn_always_enabled`                                  TODO                                                             TODO
:ref:`_assert`                                                 TODO                                                             TODO
=============================================================  ===============================================================  =======================================

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

.. _Others:

Other operations
~~~~~~~~~~~~~~~~

.. _atleast_1d:

atleast_1d
^^^^^^^^^^

.. _atleast_2d:

atleast_2d
^^^^^^^^^^

.. _atleast_3d:

atleast_3d
^^^^^^^^^^

.. _bincount:

bincount
^^^^^^^^

.. _block_diag:

block_diag
^^^^^^^^^^

.. _broadcast_tensors:

broadcast_tensors
^^^^^^^^^^^^^^^^^

.. _broadcast_to:

broadcast_to
^^^^^^^^^^^^

.. _bucketize:

bucketize
^^^^^^^^^

.. _cartesian_prod:

cartesian_prod
^^^^^^^^^^^^^^

.. _clone:

clone
^^^^^
