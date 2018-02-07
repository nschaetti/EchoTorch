# -*- coding: utf-8 -*-
#

# Imports
import torch
from .Transformer import Transformer


# Transform text to character vectors
class Character(Transformer):
    """
    Transform text to character vectors
    """

    # Constructor
    def __init__(self, alphabet):
        """
        Constructor
        :param model: Spacy's model to load.
        """
        # Properties
        self.alphabet = alphabet

        # Super constructor
        super(Character, self).__init__()
    # end __init__

    ##############################################
    # Public
    ##############################################

    # Get tags
    def get_tags(self):
        """
        Get tags.
        :return: A tag list.
        """
        return self.alphabet
    # end get_tags

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
        return len(self.get_tags()) + 1
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
        # Inputs as tensor
        inputs = torch.FloatTensor(1, self.input_dim)

        # Null symbol
        null_symbol = torch.zeros(1, self.input_dim)
        null_symbol[0, -1] = 1.0

        # Start
        start = True

        # For each character
        for c in text:
            # Replace if not function word
            if c not in self.alphabet:
                c_sym = null_symbol
            else:
                c_sym = self.tag_to_symbol(c)
            # end if

            # Add
            if not start:
                inputs = torch.cat((inputs, c_sym), dim=0)
            else:
                inputs = c_sym
                start = False
            # end if
        # end for

        return inputs
    # end convert

# end FunctionWord
