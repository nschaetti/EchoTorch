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
    def __init__(self, weights):
        """
        Constructor
        :param weights: Embedding weight matrix
        """
        # Properties
        self.weights = weights
        self.voc_size = weights.size(0)
        self.embedding_dim = weights.size(1)
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
        return self.embedding_dim
    # end input_dim

    ##############################################
    # Override
    ##############################################

    # Convert a string
    def __call__(self, idxs):
        """
        Convert a strng
        :param text:
        :return:
        """
        # Inputs as tensor
        inputs = torch.FloatTensor(1, self.embedding_dim)

        # Start
        start = True
        count = 0.0

        # OOV
        zero = 0.0
        self.oov = 0.0

        # For each inputs
        for i in range(idxs.size(0)):
            # Get token ix
            ix = idxs[i]

            # Get vector
            if ix < self.voc_size:
                embedding_vector = self.weights[ix]
            else:
                embedding_vector = torch.zeros(self.embedding_dim)
            # end if

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


# end Embedding
