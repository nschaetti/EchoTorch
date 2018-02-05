# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset
import urllib
import os
import zipfile
import json
import codecs


# Reuters C50 dataset
class ReutersC50Dataset(Dataset):
    """
    Reuters C50 dataset
    """

    # Constructor
    def __init__(self, root='./data', download=False, n_authors=50, dataset_size=100, authors=None, transform=None):
        """
        Constructor
        :param root: Data root directory.
        :param download: Download the dataset?
        :param n_authors: How many authors from the dataset to load (2 to 50).
        :param dataset_size: How many samples from each author to load (1 to 100).
        :param authors: The list of authors name to load.
        :param transform: A TextTransformer object to apply.
        """
        # Properties
        self.root = root
        self.n_authors = n_authors
        self.dataset_size = dataset_size
        self.authors = authors
        self.transform = transform
        self.author2id = dict()
        self.id2author = dict()
        self.texts = list()

        # Download the data set
        if download:
            self._download()
        # end if

        # Generate data set
        self._load()
    # end __init__

    #############################################
    # OVERRIDE
    #############################################

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return self.n_authors * self.dataset_size
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        # Current file
        text_path, author_name = self.texts[idx]

        # Read text
        text_content = codecs.open(text_path, 'r', encoding='utf-8').read()

        # Transform
        if self.transform is not None:
            return self.transform(text_content), author_name
        else:
            return text_content, author_name
        # end if
    # end __getitem__

    ##############################################
    # PRIVATE
    ##############################################

    # Download the dataset
    def _download(self):
        """
        Downlaod the dataset
        :return:
        """
        # Path to zip file
        path_to_zip = os.path.join(self.root, "reutersc50.zip")

        # Download
        urllib.urlretrieve("http://www.nilsschaetti.com/datasets/reutersc50.zip", path_to_zip)

        # Unzip
        zip_ref = zipfile.ZipFile(path_to_zip, 'r')
        zip_ref.extractall(self.root)
        zip_ref.close()

        # Delete zip
        os.remove(path_to_zip)
    # end _download

    # Load dataset
    def _load(self):
        """
        Load the dataset
        :return:
        """
        # Authors info
        authors_info = json.load(open(os.path.join(self.root, "authors.json"), 'r'))

        # For each authors
        for index, author_name in enumerate(authors_info.keys()):
            # If in the set
            if index < self.n_authors and (self.authors is None or author_name in self.authors):
                # New author
                self.author2id[author_name] = index
                self.id2author[index] = author_name

                # Add each text
                for text_index, text_name in enumerate(authors_info[author_name]):
                    if text_index < self.dataset_size:
                        # Add
                        self.texts.append((os.path.join(self.root, text_name + ".txt"), author_name))
                    # end if
                # end for
            # end if
        # end for
    # end _load

# end ReutersC50Dataset
