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
:ref:`narrow`                    :class:`TimeTensor`                      Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`nonzero`                   :class:`TimeTensor`                      This operation will destroy the time dimension, the output will then be a :class:`Tensor`.
:ref:`reshape`                   :class:`TimeTensor`                      This operation will destroy the time dimension, the output will then be a :class:`Tensor`.
:ref:`row_stack`                 :class:`Tensor` + :class:`TimeTensor`    Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`scatter`                   :class:`TimeTensor`                      Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`scatter_add`               :class:`TimeTensor`                      Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`split`                     :class:`TimeTensor`                      Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`squeeze`                   :class:`TimeTensor`                      Returns a :class:`TimeTensor` if the length of the time dimension is not 1, otherwise a :class:`Tensor` is returned.
:ref:`stack`                     :class:`Tensor` + :class:`TimeTensor`    If :attr:`dim` is less or equal to the index of the time dimension, :attr:`time_dim` is incremented by one, otherwise it is not changed.
:ref:`swapaxes`                  :class:`TimeTensor`                      See :func:`torch.transpose`
:ref:`swapdims`                  :class:`TimeTensor`                      See :func:`torch.transpose`
:ref:`t`                         :class:`TimeTensor`                      0-D timeseries are return as is. When :attr:`input ` is 1-D timeseries, the time and spatial dimensions are swaped.
:ref:`take`                      :class:`TimeTensor`                      This operation will destroy the time dimension, the output will then be a :class:`Tensor`.
:ref:`take_along_dim`            :class:`TimeTensor`                      This operation will destroy the time dimension, the output will then be a :class:`Tensor`.
:ref:`tensor_split`              :class:`TimeTensor`                      Output :class:`TimeTensor` (s) will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`tile`                      :class:`TimeTensor`                      Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`transpose`                 :class:`TimeTensor`                      If time dimension of :attr:`input` is :attr:`dim0` or :attr:`dim1`, :attr:`time_dim` is changed accordingly, otherwise it is unchanged.
:ref:`unbind`                    :class:`TimeTensor`                      Output :class:`TimeTensor` (s) will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`unsqueeze`                 :class:`TimeTensor`                      If the new dimension is before the time dimension, :attr:`time_dim` is incremented, otherwise it is unchanged.
:ref:`vsplit`                    :class:`TimeTensor`                      Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`vstack`                    :class:`Tensor` + :class:`TimeTensor`    Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`where`                     :class:`TimeTensor`                      Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
===============================  =======================================  =======================================================

Pointwise Ops
^^^^^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Outputs
=============================================================  ===============================================================  =======================================
:ref:`abs`                                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`absolute`                                                :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`acos`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`arccos`                                                  :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`acosh`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`arccosh`                                                 :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`add`                                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`addcdiv`                                                 :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`addcmul`                                                 :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`angle`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`asin`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`arcsin`                                                  :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`asinh`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`arcsinh`                                                 :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`atan`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`arctan`                                                  :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`atanh`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`arctanh`                                                 :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`atan2`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`bitwise_not`                                             :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`bitwise_and`                                             :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`bitwise_or`                                              :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`bitwise_xor`                                             :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`ceil`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`clamp`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`clip`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`conj`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`copysign`                                                :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`cos`                                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`cosh`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`deg2rad`                                                 :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`div`                                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`divide`                                                  :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`digamma`                                                 :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`erf`                                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`erfc`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`erfinv`                                                  :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`exp`                                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`exp2`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`expm1`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`fake_quantize_per_channel_affine`                        TODO                                                             TODO
:ref:`fake_quantize_per_tensor_affine`                         TODO                                                             TODO
:ref:`fix`                                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`float_power`                                             :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`floor`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`floor_divide`                                            :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`fmod`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`frac`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`frexp`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`gradient`                                                :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`imag`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`ldexp`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`lerp`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`lgamma`                                                  :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`log`                                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`log10`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`log1p`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`log2`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`logit`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`logaddexp`                                               TODO                                                             TODO
:ref:`logaddexp2`                                              TODO                                                             TODO
:ref:`logical_and`                                             TODO                                                             TODO
:ref:`logical_not`                                             TODO                                                             TODO
:ref:`logical_or`                                              TODO                                                             TODO
:ref:`logical_xor`                                             TODO                                                             TODO
:ref:`logit`                                                   TODO                                                             TODO
:ref:`hypot`                                                   TODO                                                             TODO
:ref:`i0`                                                      :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`igamma`                                                  :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`igammac`                                                 :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`mul`                                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`multiply`                                                :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
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
:ref:`block_diag`                                              :class:`Tensor` + :class:`TimeTensor`                            Returns a :class:`TimeTensor` with the index of the time dimension of the first timetensor in the list.
:ref:`broadcast_tensors`                                       :class:`Tensor` + :class:`TimeTensor`                            :class:`TimeTensor` in the ``input`` list is returned broadcasted as a :class:`TimeTensor` with same time index, :class:`Tensor` are returned broadcasted as :class:`Tensor`.
:ref:`broadcast_to`                                            :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`cartesian_prod`                                          :class:`Tensor` + :class:`TimeTensor`                            Output :class:`TimeTensor` will have an time dimension index set to 0.
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

Convolution functions
^^^^^^^^^^^^^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Outputs
=============================================================  ===============================================================  =======================================
:ref:`conv1d`                                                  :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`conv2d`                                                  :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`conv3d`                                                  :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`conv_transpose1d`                                        :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`conv_transpose2d`                                        :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`conv_transpose3d`                                        :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`unfold`                                                  :class:`TimeTensor`                                              This operation destroys the time dimension, it then returns a :class:`Tensor`.
:ref:`fold`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have a time dimension at index 2.
=============================================================  ===============================================================  =======================================

Pooling functions
^^^^^^^^^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Outputs
=============================================================  ===============================================================  =======================================
:ref:`avg_pool1d`                                              :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`avg_pool2d`                                              :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`avg_pool3d`                                              :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`max_pool1d`                                              :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`max_pool2d`                                              :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`max_pool3d`                                              :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`max_unpool1d`                                            :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`max_unpool2d`                                            :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`max_unpool3d`                                            :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`lp_pool1d`                                               :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`lp_pool2d`                                               :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`adaptive_max_pool1d`                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`adaptive_max_pool2d`                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`adaptive_max_pool3d`                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`adaptive_avg_pool1d`                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`adaptive_avg_pool2d`                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`adaptive_avg_pool3d`                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`fractional_max_pool2d`                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`fractional_max_pool3d`                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
=============================================================  ===============================================================  =======================================

Linear functions
^^^^^^^^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Outputs
=============================================================  ===============================================================  =======================================
:ref:`linear`                                                  :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`bilinear`                                                :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
=============================================================  ===============================================================  =======================================

Dropout functions
^^^^^^^^^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Outputs
=============================================================  ===============================================================  =======================================
:ref:`dropout`                                                 :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`alpha_dropout`                                           :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`feature_alpha_dropout`                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`dropout2d`                                               :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`dropout3d`                                               :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
=============================================================  ===============================================================  =======================================

Sparse functions
^^^^^^^^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Outputs
=============================================================  ===============================================================  =======================================
:ref:`embedding`                                               :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`embedding_bag`                                           :class:`TimeTensor`                                              If ``input`` is 1-D, the ``output`` is a :class:`TimeTensor` with time dimension at position 0. If 2-D, time dimension is destroyed and a :class:`Tensor` is returned.
:ref:`one_hot`                                                 :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
=============================================================  ===============================================================  =======================================

Distance functions
^^^^^^^^^^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Outputs
=============================================================  ===============================================================  =======================================
:ref:`pairwise_distance`                                       :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` if time dimension is at index 0, otherwise return a :class:`Tensor`.
:ref:`cosine_similarity`                                       :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` if ``dim`` is not equal to the index of the time dimension, otherwise return a :class:`Tensor`.
:ref:`pdist`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
=============================================================  ===============================================================  =======================================

Vision functions
^^^^^^^^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Outputs
=============================================================  ===============================================================  =======================================
:ref:`pixel_shuffle`                                           :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`pixel_unshuffle`                                         :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`pad`                                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`interpolate`                                             :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`upsample`                                                :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`upsample_nearest`                                        :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`upsample_bilinear`                                       :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`grid_sample`                                             :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
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

.. _movedim:

movedim
^^^^^^^

.. _squeeze:

squeeze
^^^^^^^

.. _tr:

t
^

.. _stack:

stack
^^^^^

.. _transpose:

transpose
^^^^^^^^^

.. _unbind:

unbind
^^^^^^

.. _unsqueeze:

unsqueeze
^^^^^^^^^

.. _Others:

Other operations
~~~~~~~~~~~~~~~~

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

.. _bucketize:

bucketize
^^^^^^^^^

.. _cartesian_prod:

cartesian_prod
^^^^^^^^^^^^^^

.. _clone:

clone
^^^^^
