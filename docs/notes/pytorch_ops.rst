Compatibility between TimeTensors and PyTorch operations
========================================================

This page list the compatibilities with the different PyTorch operations on :class:`TimeTensor`.

.. contents:: Table of Contents

.. _Summary:

Summary
~~~~~~~

Indexing, Slicing, Joining, Mutating Ops
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

===============================  =======================================  =======================================================
PyTorch Ops                      Inputs                                   Time dimension inference rule
===============================  =======================================  =======================================================
:ref:`cat`                       :class:`Tensor` + :class:`TimeTensor`    If one of the ``input`` is a :class:`TimeTensor`, returns a :class:`TimeTensor`. If one of the input is a :class:`TimeTensor`, returns a :class:`TimeTensor`.
:ref:`chunk`                     :class:`TimeTensor`                      Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`dsplit`                    :class:`TimeTensor`                      Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`column_stack`              :class:`Tensor` + :class:`TimeTensor`    Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor` in :attr:`input`.
:ref:`dstack`                    :class:`Tensor` + :class:`TimeTensor`    The index of the time dimension of a 0-D timeseries will increase from 0 to 1, otherwise it will stay the same.
:ref:`gather`                    :class:`TimeTensor`                      Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`hsplit`                    :class:`TimeTensor`                      Output :class:`TimeTensor` (s) will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`hstack`                    :class:`Tensor` + :class:`TimeTensor`    Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor` in :attr:`input`.
:ref:`index_select`              :class:`TimeTensor`                      Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`masked_select`             :class:`TimeTensor`                      This operation will destroy the time dimension, the output will then be a :class:`Tensor`.
:ref:`movedim`                   :class:`TimeTensor`                      The returned :class:`TimeTensor` will have its time dimension moved :attr:`source` or :attr:`destination` is equal to the index of the time dimension.
:ref:`moveaxis`                  :class:`TimeTensor`                      The returned :class:`TimeTensor` will have its time dimension moved :attr:`source` or :attr:`destination` is equal to the index of the time dimension.
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

Pointwise operations returns a :class:`TimeTensor` with the same index for the time dimension as the :attr:`input`. If the operation takes more than one input, the output
:class:`TimeTensor` will have the same time index as the first timetensor in the input.

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Time dimension inference rule
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
:ref:`div`                                                     :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`divide`                                                  :class:`TimeTensor` + :class:`Tensor`                            See :func:`torch.div`.
:ref:`digamma`                                                 :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`erf`                                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`erfc`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`erfinv`                                                  :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`exp`                                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`exp2`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`expm1`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`fake_quantize_per_channel_affine`                        :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`fake_quantize_per_tensor_affine`                         :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
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
:ref:`logaddexp`                                               :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`logaddexp2`                                              :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`logical_and`                                             :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`logical_not`                                             :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`logical_or`                                              :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`logical_xor`                                             :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`logit`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`hypot`                                                   :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`i0`                                                      :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`igamma`                                                  :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`igammac`                                                 :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`mul`                                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`multiply`                                                :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`mvlgamma`                                                :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`nan_to_num`                                              :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`neg`                                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`negative`                                                :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`nextafter`                                               :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`polygamma`                                               :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`positive`                                                :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`pow`                                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`rad2deg`                                                 :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`real`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`reciprocal`                                              :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`remainder`                                               :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`round`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`rsqrt`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`sigmoid`                                                 :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`sign`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`sgn`                                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`signbit`                                                 :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`sin`                                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`sinc`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`sinh`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`sqrt`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`square`                                                  :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`sub`                                                     :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`subtract`                                                :class:`TimeTensor` + :class:`Tensor`                            See :func:`torch.substract`.
:ref:`tan`                                                     :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`tanh`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`true_divide`                                             :class:`TimeTensor` + :class:`Tensor`                            Alias for :func:`div` with :attr:`rounding_mode=None`.
:ref:`trunc`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`xlogy`                                                   :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor`.
=============================================================  ===============================================================  =======================================

Reduction Ops
^^^^^^^^^^^^^

Reduction operations with a :attr:`dim` parameter will return a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor.
Indeed, if :attr:`dim` is equal to the index of the time dimension, the operation will reduce the time dimension which will then disappear and a :class:`Tensor` will be returned.

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Time dimension inference rule
=============================================================  ===============================================================  =======================================
:ref:`argmax`                                                  :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`argmin`                                                  :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`amax`                                                    :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`amin`                                                    :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`all`                                                     :class:`TimeTensor`                                              Return a ``boolean``.
:ref:`any`                                                     :class:`TimeTensor`                                              Return a ``boolean``.
:ref:`max`                                                     :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`min`                                                     :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`dist`                                                    :class:`TimeTensor`                                              This operation will destroy the time dimension, the output will then be a :class:`Tensor`.
:ref:`logsumexp`                                               :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`mean`                                                    :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`median`                                                  :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`nanmedian`                                               :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`mode`                                                    :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`norm`                                                    :class:`TimeTensor`                                              This operation will destroy the time dimension, the output will then be a :class:`Tensor`.
:ref:`nansum`                                                  :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`prod`                                                    :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`quantile`                                                :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`nanquantile`                                             :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`std`                                                     :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`std_mean`                                                :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`sum`                                                     :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`unique`                                                  :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`unique_consecutive`                                      :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`var`                                                     :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`var_mean`                                                :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`count_nonzero`                                           :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
=============================================================  ===============================================================  =======================================

Comparison Ops
^^^^^^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Time dimension inference rule
=============================================================  ===============================================================  =======================================
:ref:`allclose`                                                :class:`TimeTensor`                                              Return a ``boolean``.
:ref:`argsort`                                                 :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`eq`                                                      :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`equal`                                                   :class:`TimeTensor` + :class:`Tensor`                            Return a ``boolean``.
:ref:`ge`                                                      :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` filled with ``boolean`` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`greater_equal`                                           :class:`TimeTensor` + :class:`Tensor`                            Alias for :func:`torch.ge`.
:ref:`gt`                                                      :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` filled with ``boolean`` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`greated`                                                 :class:`TimeTensor` + :class:`Tensor`                            Alias for :func:`gt`.
:ref:`isclose`                                                 :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` filled with ``boolean`` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`isfinite`                                                :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` filled with ``boolean`` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`isinf`                                                   :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` filled with ``boolean`` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`isposinf`                                                :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` filled with ``boolean`` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`isneginf`                                                :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` filled with ``boolean`` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`isnan`                                                   :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` filled with ``boolean`` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`isreal`                                                  :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` filled with ``boolean`` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`kthvalue`                                                :class:`TimeTensor`                                              Returns a :class:`TimeTensor` if :attr:`dim` is not equal to the index of the time dimension in the input timetensor, otherwise a :class:`Tensor` is returned.
:ref:`le`                                                      :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` filled with ``boolean`` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`less_equal`                                              :class:`TimeTensor` + :class:`Tensor`                            Alias for :func:`le`.
:ref:`lt`                                                      :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` filled with ``boolean`` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`less`                                                    :class:`TimeTensor` + :class:`Tensor`                            Alias for :func:`lt`.
:ref:`maximum`                                                 :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`minimum`                                                 :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`fmax`                                                    :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`fmin`                                                    :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`ne`                                                      :class:`TimeTensor`                                              Output :class:`TimeTensor` filled with ``boolean`` will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`not_equal`                                               :class:`TimeTensor`                                              Alias for :func:`ne`.
:ref:`sort`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` (s) will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`topk`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` (s) will have the same time dimension index as the first :class:`TimeTensor`.
:ref:`msort`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` (s) will have the same time dimension index as the first :class:`TimeTensor`.
=============================================================  ===============================================================  =======================================

Spectral Ops
^^^^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Time dimension inference rule
=============================================================  ===============================================================  =======================================
:ref:`stft`                                                    :class:`TimeTensor`                                              Output :class:`TimeTensor` with :attr:`time_dim=1` if no batch dimension given, otherwise :attr:`time_dim=2`.
:ref:`istft`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` with :attr:`time_dim=0` if no batch dimension given, otherwise :attr:`time_dim=1`.
=============================================================  ===============================================================  =======================================

Other operations
^^^^^^^^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Time dimension inference rule
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
:ref:`combinations`                                            :class:`TimeTensor`                                              :class:`TimeTensor` time_dim=0
:ref:`cross`                                                   :class:`TimeTensor`                                              :class:`TimeTensor` same time_dim
:ref:`cummax`                                                  :class:`TimeTensor`                                              Returns a :class:`TimeTensor` with the index of the time dimension of the first timetensor in the list.
:ref:`cummin`                                                  :class:`TimeTensor`                                              Returns a :class:`TimeTensor` with the index of the time dimension of the first timetensor in the list.
:ref:`cumprod`                                                 :class:`TimeTensor`                                              Returns a :class:`TimeTensor` with the index of the time dimension of the first timetensor in the list.
:ref:`cumsum`                                                  :class:`TimeTensor`                                              Returns a :class:`TimeTensor` with the index of the time dimension of the first timetensor in the list.
:ref:`diag`                                                    :class:`TimeTensor`                                              This operation destroys the time dimension, it then returns a :class:`Tensor`.
:ref:`diag_embed`                                              TODO                                                             TODO
:ref:`diagflat`                                                TODO                                                             TODO
:ref:`diagonal`                                                TODO                                                             TODO
:ref:`diff`                                                    TODO                                                             TODO
:ref:`einsum`                                                  TODO                                                             TODO
:ref:`flatten`                                                 TODO                                                             This operation destroys the time dimension, it then returns a :class:`Tensor`.
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
PyTorch Ops                                                    Inputs                                                           Time dimension inference rule
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
PyTorch Ops                                                    Inputs                                                           Time dimension inference rule
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
PyTorch Ops                                                    Inputs                                                           Time dimension inference rule
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
PyTorch Ops                                                    Inputs                                                           Time dimension inference rule
=============================================================  ===============================================================  =======================================
:ref:`linear`                                                  :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`bilinear`                                                :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
=============================================================  ===============================================================  =======================================

Dropout functions
^^^^^^^^^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Time dimension inference rule
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
PyTorch Ops                                                    Inputs                                                           Time dimension inference rule
=============================================================  ===============================================================  =======================================
:ref:`embedding`                                               :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
:ref:`embedding_bag`                                           :class:`TimeTensor`                                              If ``input`` is 1-D, the ``output`` is a :class:`TimeTensor` with time dimension at position 0. If 2-D, time dimension is destroyed and a :class:`Tensor` is returned.
:ref:`one_hot`                                                 :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
=============================================================  ===============================================================  =======================================

Distance functions
^^^^^^^^^^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Time dimension inference rule
=============================================================  ===============================================================  =======================================
:ref:`pairwise_distance`                                       :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` if time dimension is at index 0, otherwise return a :class:`Tensor`.
:ref:`cosine_similarity`                                       :class:`TimeTensor` + :class:`Tensor`                            Output :class:`TimeTensor` if ``dim`` is not equal to the index of the time dimension, otherwise return a :class:`Tensor`.
:ref:`pdist`                                                   :class:`TimeTensor`                                              Output :class:`TimeTensor` will have the same time dimension index as the input :class:`TimeTensor`.
=============================================================  ===============================================================  =======================================

Vision functions
^^^^^^^^^^^^^^^^

=============================================================  ===============================================================  =======================================
PyTorch Ops                                                    Inputs                                                           Time dimension inference rule
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
