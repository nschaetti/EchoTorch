# -*- coding: utf-8 -*-
#

# Imports
import torch
from .Transformer import Transformer


# Transform text to character 3-gram
class Character3Gram(Transformer):
    """
    Transform text to character 3-grams
    """

    # Constructor
    def __init__(self, uppercase=False):
        """
        Constructor
        """
        # Properties
        self.gram_to_ix = dict()
        self.ix_to_gram = dict()
        self.gram_count = 0
        self.uppercase = uppercase

        # Super constructor
        super(Character3Gram, self).__init__()
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
        return 1
    # end input_dim

    # Vocabulary size
    @property
    def voc_size(self):
        """
        Vocabulary size
        :return:
        """
        return self.gram_count
    # end voc_size

    ##############################################
    # Private
    ##############################################

    # To upper
    def to_upper(self, gram):
        """
        To upper
        :param gram:
        :return:
        """
        if not self.uppercase:
            return gram.lower()
        # end if
        return gram
    # end to_upper

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
        # Add to voc
        for i in range(len(text)-2):
            gram = self.to_upper(text[i] + text[i+1] + text[i+2])
            if gram not in self.gram_to_ix.keys():
                self.gram_to_ix[gram] = self.gram_count
                self.ix_to_gram[self.gram_count] = gram
                self.gram_count += 1
            # end if
        # end for

        # List of character to 3 grams
        text_idxs = [self.gram_to_ix[self.to_upper(text[i] + text[i+1] + text[i+2])] for i in range(len(text)-2)]

        # To long tensor
        text_idxs = torch.LongTensor(text_idxs)

        return text_idxs, text_idxs.size(0)
    # end convert

# end Character3Gram
