# -*- coding: utf-8 -*-
#

# Imports
import gensim
from gensim.utils import tokenize
import torch


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

        # Load
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
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
        count = 0

        # For each tokens
        for token in tokenize(text):
            word_vector = self.model[token]
            if not start:
                inputs = torch.cat((inputs, torch.FloatTensor(word_vector).unsqueeze_(0)), dim=0)
            else:
                inputs = torch.FloatTensor(word_vector).unsqueeze_(0)
                start = False
            # end if
            count += 1
        # end for

        return inputs, inputs.size()[0]
    # end convert

    ##############################################
    # Static
    ##############################################

# end GensimModel
