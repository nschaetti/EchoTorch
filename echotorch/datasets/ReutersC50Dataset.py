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
from random import shuffle
import math
import pickle
from datetime import datetime


# Reuters C50 dataset
class ReutersC50Dataset(Dataset):
    """
    Reuters C50 dataset
    """

    # Constructor
    def __init__(self, root='./data', download=False, n_authors=50, dataset_size=100, dataset_start=0, authors=None,
                 transform=None, train=True, k=10, retain_transform=False, load_transform=False):
        """
        Constructor
        :param root: Data root directory.
        :param download: Download the dataset?
        :param n_authors: How many authors from the dataset to load (2 to 50).
        :param dataset_size: How many samples from each author to load (1 to 100).
        :param authors: The list of authors name to load.
        :param transform: A TextTransformer object to apply.
        :param retain_transform:
        """
        # Properties
        self.root = root
        self.n_authors = n_authors if authors is None else len(authors)
        self.dataset_size = dataset_size
        self.dataset_start = dataset_start
        self.authors = authors
        self.transform = transform
        self.author2id = dict()
        self.id2author = dict()
        self.texts = list()
        self.train = train
        self.k = k
        self.fold = 0
        self.retain_transform = retain_transform
        self.load_transform = load_transform

        # Check path
        if self.dataset_size * self.n_authors < self.k:
            self.k = self.dataset_size * self.n_authors
        # end if

        # Create directory if needed
        if not os.path.exists(self.root):
            self._create_root()
        # end if

        # Download the data set
        if download and not os.path.exists(os.path.join(self.root, "authors.json")):
            self._download()
        # end if

        # Generate data set
        self._load()
    # end __init__

    #############################################
    # PUBLIC
    #############################################

    # Set fold
    def set_fold(self, fold):
        """
        Set fold
        :param fold:
        :return:
        """
        self.fold = fold

        # Select data
        self._select_data()
    # end set_fold

    # Set train (true, false)
    def set_train(self, mode):
        """
        Set train (true, false)
        :param mode:
        :return:
        """
        self.train = mode

        # Select data
        self._select_data()
    # end set_train

    # Set start
    def set_start(self, start):
        """
        Set start
        :param start:
        :return:
        """
        self.dataset_start = start
    # end set_start

    #############################################
    # OVERRIDE
    #############################################

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return len(self.fold_texts)
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        # Current file
        text_path, author_name = self.fold_texts[idx]
        # print(text_path)
        # Read text
        text_content = codecs.open(text_path, 'r', encoding='utf-8').read()

        # Transform
        if self.transform is not None:
            # Load transform
            transformed = None
            if self.load_transform:
                transformed = self._load_transform(text_path, type(self.transform).__name__)
            # end if

            # Transform if not found
            if transformed is None:
                transformed, transformed_size = self.transform(text_content)
                to_be_saved = True
            else:
                to_be_saved = False
            # end if

            # Save transform
            if to_be_saved and self.retain_transform:
                self._save_transform(transformed, text_path, type(self.transform).__name__)
            # end if

            return transformed, self.author2id[author_name], self._create_labels(author_name, transformed_size)
        else:
            return text_content, self.author2id[author_name]
        # end if
    # end __getitem__

    ##############################################
    # PRIVATE
    ##############################################

    # Save transform
    def _save_transform(self, transform, text_path, transform_name):
        """
        Save transform
        :param text_path:
        :param transform_name:
        :return:
        """
        print(u"load")
        return pickle.dump(transform, open(text_path + "." + transform_name, 'wb'))
    # end _save_transform

    # Load transform
    def _load_transform(self, text_path, transform_name):
        """
        Load transform
        :param text_path:
        :return:
        """
        print(u"load")
        try:
            return pickle.load(open(text_path + "." + transform_name, 'rb'))
        except IOError:
            return None
        # end try
    # end if

    # Create labels
    def _create_labels(self, author_name, length):
        """
        Create labels
        :param author_name:
        :param length:
        :return:
        """
        # Author id
        author_id = self.author2id[author_name]

        # Vector
        tag_vector = torch.zeros(length, self.n_authors)

        # Set
        tag_vector[:, author_id] = 1.0

        return tag_vector
    # end _create_labels

    # Create the root directory
    def _create_root(self):
        """
        Create the root directory
        :return:
        """
        os.mkdir(self.root)
    # end _create_root

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

        # Author count
        author_count = 0

        # Given authors
        if self.authors is not None:
            given_authors = list(self.authors)
        else:
            given_authors = None
        # end if
        self.authors = list()

        # For each authors
        for index, author_name in enumerate(authors_info.keys()):
            # If in the set
            if author_count < self.n_authors and (given_authors is None or author_name in given_authors):
                # New author
                self.author2id[author_name] = author_count
                self.id2author[index] = author_name

                # Add each text
                for text_index, text_name in enumerate(authors_info[author_name]):
                    if text_index >= self.dataset_start and text_index < self.dataset_start + self.dataset_size:
                        # Add
                        self.texts.append((os.path.join(self.root, text_name + ".txt"), author_name))
                    # end if
                # end for

                # Count
                self.authors.append(author_name)
                author_count += 1
            # end if
        # end for

        # Shuffle texts
        shuffle(self.texts)

        # Select data
        self._select_data()
    # end _load

    # Select data
    def _select_data(self):
        """
        Select data
        :return:
        """
        # Fold size
        fold_quotient = math.ceil(len(self.texts) / self.k)
        fold_reste = int(math.ceil((len(self.texts) / float(self.k) - fold_quotient) * 10.0))
        fold_sizes = [fold_quotient+1]*fold_reste + [fold_quotient]*(self.k - fold_reste)

        # Fold size
        fold_size = int(fold_sizes[self.fold])

        # Compute starting point
        starting = 0
        for i in range(self.fold):
            starting += int(fold_sizes[i])
        # end for

        # Test set
        test_set = self.texts[starting:starting+fold_size]

        # Data texts
        if not self.train:
            # Test
            self.fold_texts = test_set
        else:
            # Train
            self.fold_texts = list(self.texts)

            # Remove test
            for t in test_set:
                self.fold_texts.remove(t)
            # end for
        # end if
    # end _select_data

# end ReutersC50Dataset
