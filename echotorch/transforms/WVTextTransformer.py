# -*- coding: utf-8 -*-
#

# Imports
import torch
import spacy


# Transform text to word vectors
class WVTextTransformer(object):
    """
    Transform text to word vectors
    """

    # Constructor
    def __init__(self, model="en_vectors_web_lg"):
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
        return 300
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
        inputs = torch.FloatTensor()

        # For each tokens
        for token in self.nlp(text):
            torch.cat((inputs, torch.FloatTensor(token.vector)), dimension=0)
        # end for

        return inputs
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
        return 300
    # end if

    ##############################################
    # Static
    ##############################################

# end TextTransformer
