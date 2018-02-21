# -*- coding: utf-8 -*-
#

# Imports
import torch
import spacy
from .Transformer import Transformer


# Transform text to part-of-speech vectors
class PartOfSpeech(Transformer):
    """
    Transform text to part-of-speech vectors
    """

    # Constructor
    def __init__(self, model="en_core_web_lg"):
        """
        Constructor
        :param model: Spacy's model to load.
        """
        # Super constructor
        super(PartOfSpeech, self).__init__()

        # Properties
        self.model = model
        self.nlp = spacy.load(model)
    # end __init__

    ##############################################
    # Public
    ##############################################

    # Get tags
    def get_tags(self):
        """
        Get tags.
        :return: A list of tags.
        """
        return [u"ADJ", u"ADP", u"ADV", u"CCONJ", u"DET", u"INTJ", u"NOUN", u"NUM", u"PART", u"PRON", u"PROPN",
                u"PUNCT", u"SYM", u"VERB", u"SPACE", u"X"]
    # end get_tags

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

        # For each tokens
        for token in self.nlp(text):
            pos = self.tag_to_symbol(token.pos_)

            if not start:
                inputs = torch.cat((inputs, pos), dim=0)
            else:
                inputs = pos
                start = False
            # end if
        # end for

        return inputs, inputs.size()[0]
    # end convert

# end PartOfSpeech
