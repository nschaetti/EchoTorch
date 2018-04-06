<p align="center"><img width="70%" src="docs/images/echotorch_complete.png" /></p>

--------------------------------------------------------------------------------

EchoTorch is a python module based on pyTorch to implement and test
various flavours of Echo State Network models. EchoTorch is not
intended to be put into production but for research purposes. As it is
based on pyTorch, EchoTorch's layers can be integrated into deep
architectures.
EchoTorch gives two possible ways to train models :
* Classical ESN training with Moore Penrose pseudo-inverse or LU decomposition;
* pyTorch gradient descent optimizer;

## Getting started

These instructions will get you a copy of the project up and running
on your local machine for development and testing purposes.
See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You need to following package to install EchoTorch.

* pyTorch
* TorchVision
* SpaCy
* NLTK

### Installation

EchoTorch is still under development. Until the first release you can
use the setup.py for classical installation.

## Authors

* **Nils Schaetti** - *Initial work* - [nschaetti](https://github.com/nschaetti/)

## License

This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file
for details.

## A short introduction

### Classical ESN training

You can simply create an ESN with the ESN or LiESN objects in the nn
module.

```
esn = etnn.LiESN(input_dim, n_hidden, output_dim, spectral_radius, learning_algo='inv', leaky_rate=leaky_rate)
```

Where

* input_dim is the input dimensionality;
* h_hidden is the size of the reservoir;
* output_dim is the output dimensionality;
* spectral_radius is the spectral radius with a default value of 0.9;
* learning_algo allows you to choose with training algorithms to use.
The possible values are inv, LU and sdg;

You now just have to give the ESN the inputs and the attended outputs.

```
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

```
esn.finalize()
```

The model is now trained and you can call the esn object to get a
prediction.

```
predicted = esn(test_input)
```

### ESN training with Stochastic Gradient Descent

