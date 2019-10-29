<p align="center"><img src="docs/images/echotorch_complete.png" /></p>

--------------------------------------------------------------------------------

EchoTorch is a python module based on PyTorch to implement and test
various flavours of Echo State Network models. EchoTorch is not
intended to be put into production but for research purposes. As it is
based on PyTorch, EchoTorch's layers are designed to be integrated into deep
architectures for future work and research.

<a href="https://twitter.com/intent/tweet?text=EchoTorch%20is%20a%20python%20module%20based%20on%20pyTorch%20to%20implement%20and%20test%20various%20flavours%20of%20Echo%20State%20Network%20models&url=https://github.com/nschaetti/EchoTorch&hashtags=pytorch,reservoircomputing,research">
    <img style='vertical-align: text-bottom !important;' src="https://img.shields.io/twitter/url/http/shields.io.svg?style=social" alt="Tweet">
  </a>

Join our community to create datasets and deep-learning models! Chat with us on [Gitter](https://gitter.im/EchoTorch/Lobby) and join the [Google Group](https://groups.google.com/forum/#!forum/echotorch/) to collaborate with us.

## Development status

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/echotorch.svg?style=flat-square)
[![Documentation Status](https://img.shields.io/readthedocs/echotorch/latest.svg?style=flat-square)](http://echotorch.readthedocs.io/en/latest/?badge=latest&style=flat-square)

### Builds

#### Master
[![Build Status](https://www.travis-ci.org/nschaetti/EchoTorch.svg?branch=master)](https://www.travis-ci.org/nschaetti/EchoTorch)

#### Dev
[![Build Status](https://www.travis-ci.org/nschaetti/EchoTorch.svg?branch=dev)](https://www.travis-ci.org/nschaetti/EchoTorch)

## Index

This repository consists of:

* [echotorch.datasets] : Pre-built datasets for common ESN tasks.
* echotorch.evaluation : Tools and functions to evaluate and compare ESN models.
* echotorch.models : Ready to train models and generic pre-trained ESN models.
* echotorch.nn : All neural network Torch components for ESN and Reservoir Computing.
* echotorch.transforms : Data transformations specific to ESN.
* echotorch.utils : Tools, functions and measures for ESN and Reservoir Computing.

## Examples

Here is some examples of what you can do with EchoTorch.

* [Conceptors](https://github.com/nschaetti/EchoTorch/tree/dev/examples/conceptors)
    * [Four patterns generation with Conceptors](https://github.com/nschaetti/EchoTorch/blob/dev/examples/conceptors/conceptors_4_patterns_generation.py) : load into a reservoir four patterns and re-generate them with conceptor-based neural filtering.
* [Datasets](https://github.com/nschaetti/EchoTorch/tree/dev/examples/datasets)
    * [Logistic Map](https://github.com/nschaetti/EchoTorch/blob/dev/examples/datasets/logistic_map.py) : generate data from the logistic map function.
* [Evaluation](https://github.com/nschaetti/EchoTorch/tree/dev/examples/evaluation)
    * [Fold cross-validation](https://github.com/nschaetti/EchoTorch/tree/dev/examples/evaluation/fold_cross_validation.py) : how to perform 10-fold cross validation.
    * [Grid search](https://github.com/nschaetti/EchoTorch/tree/dev/examples/evaluation/grid_search.py) : hyperparamaters optimization with grid-search.
* [Generation](https://github.com/nschaetti/EchoTorch/tree/dev/examples/generation)
    * [NARMA-10 generation with feedbacks](https://github.com/nschaetti/EchoTorch/blob/dev/examples/generation/narma10_esn_feedbacks.py) : generate NARMA-10 timeseries with feedbacks.
* [Memory](https://github.com/nschaetti/EchoTorch/tree/dev/examples/memory)
    * [Memtest](https://github.com/nschaetti/EchoTorch/blob/dev/examples/memory/memtest.py) : test the capacity of an ESN to memorize random inputs.
* [MNIST](https://github.com/nschaetti/EchoTorch/blob/dev/examples/MNIST/)
    * [Image to timeseries conversion](https://github.com/nschaetti/EchoTorch/blob/dev/examples/MNIST/convert_images.py) : how to convert images to timeseries.
* [Nodes](https://github.com/nschaetti/EchoTorch/blob/dev/examples/nodes)
    * [Independent Component Analysis](https://github.com/nschaetti/EchoTorch/blob/dev/examples/nodes/ica_tests.py) : how to do Independent Component Analysis (ICA) with EchoTorch.
    * [Principal Component Analysis](https://github.com/nschaetti/EchoTorch/blob/dev/examples/nodes/pca_tests.py) : how to do Principal Component Analysis (PCA) with EchoTorch.
    * [Slow Feature Analysis](https://github.com/nschaetti/EchoTorch/blob/dev/examples/nodes/sfa_tests.py) : how to do Slow Features Analysis (SFA) with EchoTorch.
* [Switch between attractors](https://github.com/nschaetti/EchoTorch/blob/dev/examples/switch_attractor/switch_attractor_esn.py)
    * [Switch Attractor](https://github.com/nschaetti/EchoTorch/blob/dev/examples/switch_attractor/switch_attractor_esn.py) : test the capacity of a simple ESN to switch between attractors.
* [Timeseries prediction](https://github.com/nschaetti/EchoTorch/tree/dev/examples/timeserie_prediction)
    * [Mackey Glass](https://github.com/nschaetti/EchoTorch/blob/dev/examples/timeserie_prediction/mackey_glass_esn.py) : Mackey-Glass timeseries prediction with ESN.
    * [NARMA-10](https://github.com/nschaetti/EchoTorch/blob/dev/examples/timeserie_prediction/mackey_glass_esn.py) : NARMA-10 timeseries prediction with ESN and original training methods (ridge regression).
    * [NARMA-10 for reservoir sizes](https://github.com/nschaetti/EchoTorch/blob/dev/examples/timeserie_prediction/mackey_glass_esn.py) : NARMA-10 timeseries prediction with ESN and different reservoir sizes.
    * [NARMA-10 with gradient descent](https://github.com/nschaetti/EchoTorch/blob/dev/examples/timeserie_prediction/mackey_glass_esn.py) : NARMA-10 timeseries prediction with ESN and gradient descent (it doesn't work, see tutorials).
    * [NARMA-10 with Gated-ESN](https://github.com/nschaetti/EchoTorch/blob/dev/examples/timeserie_prediction/narma10_gated_esn.py) : NARMA-10 prediction with Gated-ESN (ESN + PCA + LSTM).
    * [NARMA-10 with Stacked-ESN](https://github.com/nschaetti/EchoTorch/blob/dev/examples/timeserie_prediction/narma10_stacked_esn.py) : NARMA-10 prediction with Stacked-ESN.
* [Unsupervised Learning](https://github.com/nschaetti/EchoTorch/tree/dev/examples/unsupervised_learning)
    
## Tutorials

In addition to examples, here are some Jupyter tutorials to learn how Reservoir Computing works.

## Code and papers

Here are some experimences done with ESN and reproduced with EchoTorch :

* [Echo State Networks-Based Reservoir Computing for MNIST Handwritten Digits Recognition](https://www.researchgate.net/publication/309033779_Echo_State_Networks-Based_Reservoir_Computing_for_MNIST_Handwritten_Digits_Recognition?_sg=8VjQVy9bx8MPtY_4yrKU7xk8FXFP2hsPO9VjaBOtWgZfrgC8UJ7jEcn8xQsmM4I5-i6UKy8-41NH4Q.56KECTRGged0v4XjR9CWZveO3MoY8-ZLPxF8V9rSezvvSIcyuUtSUy9sNDI-7l7dmsnnPDir1MhG5wVvbqUDrQ&_sgd%5Bnc%5D=2&_sgd%5Bncwor%5D=0)
* Controlling RNNs by Conceptors (Herbert Jaeger) :

## Getting started

These instructions will get you a copy of the project up and running
on your local machine for development and testing purposes.
See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You need to following package to install EchoTorch.

* sphinx_bootstrap_theme
* future
* numpy
* scipy
* scikit-learn
* matplotlib
* torch==1.3.0
* torchvision==0.4.1

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

