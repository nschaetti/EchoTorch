# -*- coding: utf-8 -*-
#

# Imports
import gensim
from gensim.utils import tokenize
import torch
import numpy as np


# Transform text to vectors with embedding
class Embedding(object):
    """
    Transform text to vectors with embedding
    """

    # Constructor
    def __init__(self, embedding, size):
        """
        Constructor
        :param embedding: Embedding
        """
        # Properties
        self.embedding = embedding
        self.size = size
    # end __init__

    ##############################################
    # Properties
    ##############################################

    # Get the number of inputs
    @property
    def input_dim(self):
        """
        Get the number of inputs
        :return:
        """
        return self.size
    # end input_dim

    ##############################################
    # Override
    ##############################################

    # Convert a string
    def __call__(self, text_array):
        """
        Convert a strng
        :param text:
        :return:
        """
        # Inputs as tensor
        inputs = torch.FloatTensor(1, self.input_dim)

        # Start
        start = True
        count = 0.0

        # OOV
        zero = 0.0
        self.oov = 0.0

        # For each tokens
        for token in text_array:
            # Get vector
            embedding_vector = self.embedding(token)

            # Test zero
            if torch.sum(embedding_vector) == 0.0:
                zero += 1.0
                embedding_vector = np.zeros(self.input_dim)
            # end if

            # Start/continue
            if not start:
                inputs = torch.cat((inputs, torch.FloatTensor(embedding_vector).unsqueeze_(0)), dim=0)
            else:
                inputs = torch.FloatTensor(embedding_vector).unsqueeze_(0)
                start = False
            # end if
            count += 1
        # end for

        # OOV
        self.oov = zero / count * 100.0

        return inputs, inputs.size()[0]
    # end convert

    ##############################################
    # Static
    ##############################################


# end GloveVector
