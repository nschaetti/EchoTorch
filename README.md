<p align="center"><img src="docs/images/echotorch_complete.png" /></p>

--------------------------------------------------------------------------------

EchoTorch is a python module based on pyTorch to implement and test
various flavours of Echo State Network models. EchoTorch is not
intended to be put into production but for research purposes. As it is
based on pyTorch, EchoTorch's layers can be integrated into deep
architectures.
EchoTorch gives two possible ways to train models :
* Classical ESN training with Moore Penrose pseudo-inverse or LU decomposition;
* pyTorch gradient descent optimizer;

<a href="https://twitter.com/intent/tweet?text=EchoTorch%20is%20a%20python%20module%20based%20on%20pyTorch%20to%20implement%20and%20test%20various%20flavours%20of%20Echo%20State%20Network%20models&url=https://github.com/nschaetti/EchoTorch&hashtags=pytorch,reservoircomputing,research">
    <img style='vertical-align: text-bottom !important;' src="https://img.shields.io/twitter/url/http/shields.io.svg?style=social" alt="Tweet">
  </a>

Join our community to create datasets and deep-learning models! Chat with us on [Gitter](https://gitter.im/EchoTorch/Lobby) and join the [Google Group](https://groups.google.com/forum/#!forum/echotorch/) to collaborate with us.

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/echotorch.svg?style=flat-square)
[![Codecov](https://img.shields.io/codecov/c/github/nschaetti/echotorch/master.svg?style=flat-square)](https://codecov.io/gh/nschaetti/EchoTorch)
[![Documentation Status](	https://img.shields.io/readthedocs/echotorch/latest.svg?style=flat-square)](http://echotorch.readthedocs.io/en/latest/?badge=latest&style=flat-square)
[![Build Status](https://img.shields.io/travis/nschaetti/EchoTorch/master.svg?style=flat-square)](https://travis-ci.org/nschaetti/EchoTorch)

This repository consists of:

* echotorch.datasets : Pre-built datasets for common ESN tasks
* echotorch.models : Generic pretrained ESN models
* echotorch.transforms : Data transformations specific to echo state networks
* echotorch.utils : Tools, functions and measures for echo state networks

## Getting started

These instructions will get you a copy of the project up and running
on your local machine for development and testing purposes.
See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You need to following package to install EchoTorch.

* pyTorch
* TorchVision

### Installation

    pip install EchoTorch

## Authors

* **Nils Schaetti** - *Initial work* - [nschaetti](https://github.com/nschaetti/)

## License

This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file
for details.

## Citing

If you find EchoTorch useful for an academic publication, then please use the following BibTeX to cite it:

```
@misc{echotorch,
  author = {Schaetti, Nils},
  title = {EchoTorch: Reservoir Computing with pyTorch},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nschaetti/EchoTorch}},
}
```

## A short introduction

### Classical ESN training

You can simply create an ESN with the ESN or LiESN objects in the nn
module.

```python
esn = etnn.LiESN(
    input_dim,
    n_hidden,
    output_dim,
    spectral_radius,
    learning_algo='inv',
    leaky_rate=leaky_rate
)
```

Where

* input_dim is the input dimensionality;
* h_hidden is the size of the reservoir;
* output_dim is the output dimensionality;
* spectral_radius is the spectral radius with a default value of 0.9;
* learning_algo allows you to choose with training algorithms to use.
The possible values are inv, LU and sdg;

You now just have to give the ESN the inputs and the attended outputs.

```python
for data in trainloader:
    # Inputs and outputs
    inputs, targets = data

    # To variable
    inputs, targets = Variable(inputs), Variable(targets)

    # Give the example to EchoTorch
    esn(inputs, targets)
# end for
```

After giving all examples to EchoTorch, you just have to call the
finalize method.

```python
esn.finalize()
```

The model is now trained and you can call the esn object to get a
prediction.

```python
predicted = esn(test_input)
```

### ESN training with Stochastic Gradient Descent

