.. EchoTorch documentation master file, created by
   sphinx-quickstart on Thu Apr  6 11:30:46 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

EchoTorch documentation
=======================

EchoTorch is an PyTorch-based library for Reservoir Computing (RC) and Echo State Network (ESN) using GPUs and CPUs. It
is designed to simplify the evaluation and implementation of machine learning models based on ESNs and RC, but also to
be a user-friendly time series analysis tool for researchers and engineers. EchoTorch is based on a fully modular
architecture which allow developers to implement and evaluate quickly different RC architectures on timeseries tasks.

It also allows to implement more advanced models based on Conceptors and on the Deep Learning paradigm such as
DeepESN. You will also be able to integrate random recurring models into your PyTorch models and simply
generate time series data for your research.

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Notes

    notes/*

.. toctree::
    :maxdepth: 1
    :caption: Python API

    echotorch
    timetensors
    echotorch.data
    echotorch.models
    echotorch.nn
    echotorch.skecho
    echotorch.transforms
    echotorch.utils
    echotorch.viz
    echotorch.acf
    time_tensors
    basetensors

.. toctree::
    :maxdepth: 1
    :caption: Libraries

    torchaudio <https://pytorch.org/audio/stable>
    torchtext <https://pytorch.org/text/stable>
    torchvision <https://pytorch.org/vision/stable>
    TorchServe <https://pytorch.org/serve>
    PyTorch on XLA Devices <http://pytorch.org/xla/>


.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Community

    PyTorch Contribution Guide (apply to EchoTorch) <https://pytorch.org/docs/stable/community/contribution_guide.html>
    community/*

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _pytorch: http://pytorch.org/
.. _sklearn: http://scikit-learn.org/
