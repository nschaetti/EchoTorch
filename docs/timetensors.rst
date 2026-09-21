.. currentmodule:: echotorch

.. _tensor-doc:

echotorch.TimeTensor
===================================

TimeTensor class reference
--------------------------

A :class:`TimeTensor` wraps a PyTorch :class:`torch.Tensor` and adds a notion of
**time dimension** (``time_dim``).  All time-based operations of EchoTorch
(statistical ops, concatenation, indexing, ...) rely on this class.

.. autoclass:: echotorch.TimeTensor
    :members:
    :inherited-members:
    :show-inheritance:
    :special-members: __init__, __getitem__, __setitem__, __len__, __repr__,
        __lt__, __le__, __gt__, __ge__, __eq__, __ne__

Constructors
------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    TimeTensor.new_timetensor
    TimeTensor.new_timetensor_with_func

Properties
----------

The wrapped tensor, the position of the time dimension and derived sizes.

.. autosummary::
    :toctree: generated
    :nosignatures:

    TimeTensor.tensor
    TimeTensor.time_dim
    TimeTensor.tlen
    TimeTensor.cdim
    TimeTensor.bdim
    TimeTensor.csize
    TimeTensor.bsize
    TimeTensor.numelc
    TimeTensor.numelb

Casting
-------

.. autosummary::
    :toctree: generated
    :nosignatures:

    TimeTensor.to

Indexing, Slicing
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    TimeTensor.indexing_timetensor
    TimeTensor.as_strided

Dimension manipulation (after- callbacks)
------------------------------------------

The following ``after_*`` methods are used internally by
:func:`TimeTensor.__torch_function__` to re-wrap the result of a torch
operation (or of a sequence of torch operations) into a :class:`TimeTensor`
with the correct time dimension.

.. autosummary::
    :toctree: generated
    :nosignatures:

    TimeTensor.after_dstack
    TimeTensor.after_movedim
    TimeTensor.after_squeeze
    TimeTensor.after_stack
    TimeTensor.after_t
    TimeTensor.after_transpose
    TimeTensor.after_permute
    TimeTensor.after_unbind
    TimeTensor.after_unsqueeze
    TimeTensor.after_flatten
    TimeTensor.after_reshape
    TimeTensor.after_vstack
    TimeTensor.after_atleast_2d
    TimeTensor.after_atleast_3d
    TimeTensor.after_broadcast_tensors
    TimeTensor.after_cartesian_prod
    TimeTensor.after_cdist
    TimeTensor.after_combinations
    TimeTensor.after_cummax
    TimeTensor.after_cummin
    TimeTensor.after_kthvalue
    TimeTensor.after_topk
    TimeTensor.after_fold
    TimeTensor.after_embedding_bag
    TimeTensor.after_pairwise_distance

Matrix and spectral operations
------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    TimeTensor.mm

Spectral-domain callbacks
-------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    TimeTensor.after_stft
    TimeTensor.after_istft

Conversion helpers
------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    TimeTensor.transform_to_timetensors
    TimeTensor.transform_object_to_timetensor
    TimeTensor.convert_to_timetensor
    TimeTensor.transform_similar_tensors
    TimeTensor.convert_similar_tensors
    TimeTensor.convert_after_reduction

Reduction / torch function machinery
------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    TimeTensor.check_time_dim
    TimeTensor.__torch_function__

Comparison operators
--------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    TimeTensor.__lt__
    TimeTensor.__le__
    TimeTensor.__gt__
    TimeTensor.__ge__
    TimeTensor.__eq__
    TimeTensor.__ne__
