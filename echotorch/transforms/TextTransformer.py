# -*- coding: utf-8 -*-
#

# Imports
import torch


# Base class for text transformers
class TextTransformer(object):
    """
    Base class for text transformers
    """

    # Constructor
    def __init__(self):
        """
        Constructor
        """
        pass
    # end __init__

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
        pass
    # end input_dim

    ##############################################
    # Override
    ##############################################

    # Convert a string
    def __call__(self, tokens):
        """
        Convert a string to a ESN input
        :param tokens: Text to convert
        :return: A list of symbols
        """
        pass
    # end convert

    ##############################################
    # Private
    ##############################################

    # Get inputs size
    def _get_inputs_size(self):
        """
        Get inputs size.
        :return:
        """
        return 0
    # end if

    ##############################################
    # Static
    ##############################################

# end TextTransformer
