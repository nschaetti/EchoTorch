# -*- coding: utf-8 -*-
#

# Imports
import torch
import spacy
from .Transformer import Transformer


# Transform text to a function word vectors
class FunctionWord(Transformer):
    """
    Transform text to character vectors
    """

    # Constructor
    def __init__(self, model="en_core_web_lg"):
        """
        Constructor
        :param model: Spacy's model to load.
        """
        # Super constructor
        super(FunctionWord, self).__init__()

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
        :return: A tag list.
        """
        return [u"a", u"about", u"above", u"after", u"after", u"again", u"against", u"ago", u"ahead",
                u"all",
                u"almost", u"along", u"already", u"also", u"although", u"always", u"am", u"among", u"an",
                u"and", u"any", u"are", u"aren't", u"around", u"as", u"at", u"away", u"backward",
                u"backwards", u"be", u"because", u"before", u"behind", u"below", u"beneath", u"beside",
                u"between", u"both", u"but", u"by", u"can", u"cannot", u"can't", u"cause", u"'cos",
                u"could",
                u"couldn't", u"'d", u"despite", u"did", u"didn't", u"do", u"does", u"doesn't", u"don't",
                u"down", u"during", u"each", u"either", u"even", u"ever", u"every", u"except", u"for",
                u"forward", u"from", u"had", u"hadn't", u"has", u"hasn't", u"have", u"haven't", u"he",
                u"her", u"here", u"hers", u"herself", u"him", u"himself", u"his", u"how", u"however",
                u"I",
                u"if", u"in", u"inside", u"inspite", u"instead", u"into", u"is", u"isn't", u"it", u"its",
                u"itself", u"just", u"'ll", u"least", u"less", u"like", u"'m", u"many", u"may",
                u"mayn't",
                u"me", u"might", u"mightn't", u"mine", u"more", u"most", u"much", u"must", u"mustn't",
                u"my", u"myself", u"near", u"need", u"needn't", u"needs", u"neither", u"never", u"no",
                u"none", u"nor", u"not", u"now", u"of", u"off", u"often", u"on", u"once", u"only",
                u"onto",
                u"or", u"ought", u"oughtn't", u"our", u"ours", u"ourselves", u"out", u"outside", u"over",
                u"past", u"perhaps", u"quite", u"'re", u"rather", u"'s", u"seldom", u"several", u"shall",
                u"shan't", u"she", u"should", u"shouldn't", u"since", u"so", u"some", u"sometimes",
                u"soon",
                u"than", u"that", u"the", u"their", u"theirs", u"them", u"themselves", u"then", u"there",
                u"therefore", u"these", u"they", u"this", u"those", u"though", u"through", u"thus",
                u"till",
                u"to", u"together", u"too", u"towards", u"under", u"unless", u"until", u"up", u"upon",
                u"us", u"used", u"usedn't", u"usen't", u"usually", u"'ve", u"very", u"was", u"wasn't",
                u"we", u"well", u"were", u"weren't", u"what", u"when", u"where", u"whether", u"which",
                u"while", u"who", u"whom", u"whose", u"why", u"will", u"with", u"without", u"won't",
                u"would", u"wouldn't", u"yet", u"you", u"your", u"yours", u"yourself", u"yourselves", u"X"]
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

        # Null symbol
        null_symbol = torch.zeros(1, self.input_dim)
        null_symbol[0, -1] = 1.0

        # Start
        start = True

        # For each tokens
        for token in self.nlp(text):
            # Replace if not function word
            if token.text not in self.symbols:
                token_fw = u"X"
            else:
                token_fw = token.text
            # end if

            # Get tag
            fw = self.tag_to_symbol(token_fw)

            # Add
            if not start:
                inputs = torch.cat((inputs, fw), dim=0)
            else:
                inputs = fw
                start = False
            # end if
        # end for

        return inputs, inputs.size()[0]
    # end convert

# end FunctionWord
