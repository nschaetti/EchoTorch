# -*- coding: utf-8 -*-
#

# Imports
import gensim
from gensim.utils import tokenize
import torch
import numpy as np


# Transform text to vectors with a Gensim model
class GensimModel(object):
    """
    Transform text to vectors with a Gensim model
    """

    # Constructor
    def __init__(self, model_path):
        """
        Constructor
        :param model_path: Model's path.
        """
        # Properties
        self.model_path = model_path

        # Format
        binary = False if model_path[-4:] == ".vec" else True

        # Load
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=binary, unicode_errors='ignore')

        # OOV
        self.oov = 0.0
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
        inputs = torch.FloatTensor(1, self.input_dim)

        # Start
        start = True
        count = 0.0

        # OOV
        zero = 0.0
        self.oov = 0.0

        # For each tokens
        for token in tokenize(text):
            found = False
            # Try normal
            try:
                word_vector = self.model[token]
                found = True
            except KeyError:
                pass
            # end try

            # Try lower
            if not found:
                try:
                    word_vector = self.model[token.lower()]
                except KeyError:
                    zero += 1.0
                    word_vector = np.zeros(self.input_dim)
                # end try
            # end if

            # Start/continue
            if not start:
                inputs = torch.cat((inputs, torch.FloatTensor(word_vector).unsqueeze_(0)), dim=0)
            else:
                inputs = torch.FloatTensor(word_vector).unsqueeze_(0)
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
    #########################################