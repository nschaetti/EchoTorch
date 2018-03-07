# -*- coding: utf-8 -*-
#

# Imports
from .Transformer import Transformer


# Compose multiple transformations
class Compose(Transformer):
    """
    Compose multiple transformations
    """

    # Constructor
    def __init__(self, transforms):
        """
        Constructor
        """
        # Properties
        self.transforms = transforms

        # Super constructor
        super(Compose, self).__init__()
    # end __init__

    ##############################################
    # Public
    ##############################################

    ##############################################
    # Properties
    ##############################################

    # Get the number of inputs
    @property
    def input_dim(self):
        """
        Get the number of inputs.
        :return: The input size.
        """
        return self.transforms[-1].input_dim
    # end input_dim

    ##############################################
    # Override
    ##############################################

    # Convert a string
    def __call__(self, text):
        """
        Convert a string to a ESN input
        :param text: Text to convert
        :return: Tensor of word vectors
        """
        # For each transform
        for index, transform in enumerate(self.transforms):
            # Transform
            if index == 0:
                outputs, size = transform(text)
            else:
                outputs, size = transform(outputs)
            # end if
        # end for

        return outputs, size
    # end convert

# end Compose
