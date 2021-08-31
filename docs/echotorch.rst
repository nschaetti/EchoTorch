echotorch package
=================
The echotorch main package contains data structures for timeseries and time-related tensors and
supply specific mathematical and programming operations for these specific tensors. As for PyTorch,
CUDA counterparts are given for you to run your experiments on NVIDIA GPUs.

.. currentmodule:: echotorch

.. _timetensor-base:

TimeTensors
-----------
.. autosummary::
    :toctree: generated
    :nosignatures:

    is_timetensor
    tcat

.. _timetensor-creation-ops:

Creation Ops
~~~~~~~~~~~~

.. note::
    Similarly to the PyTorch documentation, creation ops for random sampling are listed under :ref:`random-sampling`
    and include the time version of PyTorch random sampling ops:
    :func:`echotorch.rand`
    :func:`echotorch.rand_like`
    :func:`echotorch.randn`
    :func:`echotorch.randn_like`
    :func:`echotorch.randint`
    :func:`echotorch.randint_like`
    :func:`echotorch.randperm`

.. autosummary::
    :toctree: generated
    :nosignatures:

    timetensor
    as_timetensor
    sparse_coo_timetensor
    as_strided
    from_numpy
    zeros
    zeros_like
    ones
    ones_like
    arange
    linspace
    logspace
    empty
    empty_like
    empty_strided
    full
    full_like
    quantize_per_timetensor
    quantize_per_channel
    dequantize
    complex
    polar


Indexing, Slicing, Joining, Mutating Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    tcat
    tchunk
    tdsplit
    channel_stack
    cstack
    batch_stack
    bstack
    time_stack
    tstack
    tindex_select
    tmasked_select
    movecdim
    movebdim
    movetdim
    tnarrow
    creshape
    breshape
    tsplit
    csplit
    bsplit
    tt
    bt
    ct
    take_along_time
    timetensor_split
    ttile
    ctile
    btile
    transpose
    btranspose
    ctranspose
    unbind
    tunbind
    tselect

Random sampling
~~~~~~~~~~~~~~~

.. note:
    Random sampling creation ops in this section use the same corresponding ops from PyTorch but return timetensors
    with additional parameters.

.. autosummary::
    :toctree: generated
    :nosignatures:

    bernoulli
    multinomial
    normal
    poisson
    rand
    rand_like
    randint
    randint_like
    randn
    randn_like
    randperm

In-place random sampling
~~~~~~~~~~~~~~~~~~~~~~~~


Statistical Operations
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    tmean
    tstd
    tvar
    cov
    cor
    autocovariance_coeffs
