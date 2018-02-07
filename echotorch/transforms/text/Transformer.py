# -*- coding: utf-8 -*-
#

# Imports
import torch


# Base class for text transformers
class Transformer(object):
    """
    Base class for text transformers
    """

    # Constructor
    def __init__(self):
        """
        Constructor
        """
        # Properties
        self.symbols = self.generate_symbols()
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
        return len(self.get_tags())
    # end input_dim

    ##############################################
    # Public
    ##############################################

    # Get tags
    def get_tags(self):
        """
        Get tags.
        :return: A list of tags.
        """
        return []
    # end get_tags

    # Get symbol from tag
    def tag_to_symbol(self, tag):
        """
        Get symbol from tag.
        :param tag: Tag.
        :return: The corresponding symbols.
        """
        if tag in self.symbols.keys():
            return self.symbols[tag]
        return None
    # end word_to_symbol

    # Generate symbols
    def generate_symbols(self):
        """
        Generate word symbols.
        :return: Dictionary of tag to symbols.
        """
        result = dict()
        for index, p in enumerate(self.get_tags()):
            result[p] = torch.zeros(1, self.input_dim)
            result[p][0, index] = 1.0
        # end for
        return result
    # end generate_symbols

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
    # Static
    ##############################################

# end TextTransformer
