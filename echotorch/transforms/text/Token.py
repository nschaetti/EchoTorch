# -*- coding: utf-8 -*-
#

# Imports
import spacy


# Transform text to a list of tokens
class Token(object):
    """
    Transform text to a list of tokens
    """

    # Constructor
    def __init__(self, model="en_core_web_lg"):
        """
        Constructor
        :param model: Spacy's model to load.
        """
        # Properties
        self.model = model
        self.nlp = spacy.load(model)
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
        return 1
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
        # Inputs as a list
        tokens = list()

        # For each tokens
        for token in self.nlp(text):
            tokens.append(unicode(token.text))
        # end for

        return tokens, len(tokens)
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
        return 1
    # end if

    ##############################################
    # Static
    ##############################################

# end Token
