# -*- coding: utf-8 -*-
#

# Imports
import torch
from .Transformer import Transformer
import numpy as np


# Transform text to character 2-gram
class Character2Gram(Transformer):
    """
    Transform text to character 2-grams
    """

    # Constructor
    def __init__(self, uppercase=False, gram_to_ix=None, start_ix=0, fixed_length=-1, overlapse=True):
        """
        Constructor
        """
        # Gram to ix
        if gram_to_ix is not None:
            self.gram_count = len(gram_to_ix.keys())
            self.gram_to_ix = gram_to_ix
        else:
            self.gram_count = start_ix
            self.gram_to_ix = dict()
        # end if

        # Ix to gram
        self.ix_to_gram = dict()
        if gram_to_ix is not None:
            for gram in gram_to_ix.keys():
                self.ix_to_gram[gram_to_ix[gram]] = gram
            # end for
        # end if

        # Properties
        self.uppercase = uppercase
        self.fixed_length = fixed_length
        self.overlapse = overlapse

        # Super constructor
        super(Character2Gram, self).__init__()
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
        # Step
        if self.overlapse:
            step = 1
        else:
            step = 2
        #  end if

        # Add to voc
        for i in np.arange(0, len(text) - 1, step):
            gram = self.to_upper(text[i] + text[i+1])
            if gram not in self.gram_to_ix.keys():
                self.gram_to_ix[gram] = self.gram_count
                self.ix_to_gram[self.gram_count] = gram
                self.gram_count += 1
            # end if
        # end for

        # List of character to 2grams
        text_idxs = [self.gram_to_ix[self.to_upper(text[i] + text[i+1])] for i in range(len(text)-1)]

        # To long tensor
        text_idxs = torch.LongTensor(text_idxs)

        # Check length
        if self.fixed_length != -1:
            if text_idxs.size(0) > self.fixed_length:
                text_idxs = text_idxs[:self.fixed_length]
            elif text_idxs.size(0) < self.fixed_length:
                zero_idxs = torch.LongTensor(self.fixed_length).fill_(0)
                zero_idxs[:text_idxs.size(0)] = text_idxs
                text_idxs = zero_idxs
            # end if
        # end if

        return text_idxs, text_idxs.size(0)
    # end convert

# end Character2Gram
