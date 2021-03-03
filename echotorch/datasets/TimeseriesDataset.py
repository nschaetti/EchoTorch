# -*- coding: utf-8 -*-
#
# File : echotorch/datasets/TimeseriesDataset.py
# Description : Load timeseries from a directory
# Date : 23th of February, 2021
#
# This file is part of EchoTorch.  EchoTorch is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Nils Schaetti <nils.schaetti@unine.ch>


# Imports
import os
import torch
import json
import numpy as np

# echotorch imports
from echotorch.transforms import Transformer

# Imports local
from .EchoDataset import EchoDataset


# Filenames
INFO_DATA_FILE_OUTPUT = "{:07d}TS.pth"
PROPERTIES_FILE = "dataset_properties.json"
INFO_METADATA_FILE_OUTPUT = "{:07d}TS.json"


# A dataset to load timeseries from a directory with meta-data
class TimeseriesDataset(EchoDataset):
    """
    A dataset to load timeseries from a directory with meta-data
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, root_directory, transform=None, timestep=1.0, in_memory=False):
        """
        Constructor
        :param root_directory: Base root directory
        :param transform: An EchoTorch transformer
        :param timestep: The time step of time series (default: 1 second)
        """
        # Properties
        self._root_directory = root_directory
        self._transform = transform
        self._timestep = timestep
        self._root_json_file = os.path.join(self._root_directory, PROPERTIES_FILE)
        self._in_memory = False

        # Load JSON file
        self._dataset_properties = self._filter_dataset(
            self._load_properties_file(self._root_json_file)
        )

        # Load in memory if necessary
        self._sample_in_memory = False
        if in_memory:
            self._loaded_samples = self._load_samples()
        # end if
    # end __init__

    # endregion CONSTRUCTORS

    # region PROPERTIES

    # Root directory (GET)
    @property
    def root_directory(self) -> str:
        """
        Root directory
        :return: Root directory
        """
        return self._root_directory
    # end root_directory

    # Transform (GET)
    @property
    def transform(self) -> Transformer:
        """
        Transform
        :return: Transformer
        """
        return self._transform
    # end transform

    # Transform (SET)
    @transform.setter
    def transform(self, value: Transformer):
        """
        Transformer (SET)
        """
        self._transform = value
    # end transform

    # Number of samples in the dataset
    @property
    def n_samples(self) -> int:
        """
        Number of samples in the dataset
        :return: Number of samples
        """
        return self._dataset_properties['n_samples']
    # end n_samples

    # Sample length
    @property
    def sample_length(self) -> int:
        """
        Accumulated lengths ot the sample
        """
        return self._dataset_properties['sample_length']
    # end sample_length

    # Sample properties
    @property
    def sample_properties(self) -> list:
        """
        Sample properties
        """
        return self._dataset_properties['samples']
    # end sample_properties

    # Columns
    @property
    def columns(self):
        """
        Columns
        """
        return self._dataset_properties['column_names']
    # end columns

    # Columns propertiesS
    @property
    def columns_properties(self):
        """
        Columns properties
        """
        return self._dataset_properties['columns']
    # end columns_properties

    # Labels
    @property
    def labels(self) -> list:
        """
        Labels
        """
        return self._dataset_properties['labels']
    # end labels

    # Metadata
    @property
    def metadata(self) -> dict:
        """
        Metadata
        """
        return self._dataset_properties['metadata']
    # end metadata

    # Event type indices
    @property
    def event_type_indices(self) -> dict:
        """
        Event type indices
        """
        return self._dataset_properties['event_type_indices']
    # end event_type_indices

    # Event type names
    @property
    def event_type_names(self) -> list:
        """
        Event type names
        """
        return self._dataset_properties['event_type_names'].keys()
    # end event_type_names

    # Event types
    @property
    def event_types(self) -> list:
        """
        Event types
        """
        return self._dataset_properties['event_types']
    # end event_types

    # endregion PROPERTIES

    # region PUBLIC

    # Label name to index
    def label_name_to_index(self, label_name: str) -> int:
        """
        label name
        """
        return self._dataset_properties['label_names'][label_name]
    # end label_name_to_index

    # Label index to name
    def label_index_to_name(self, label_index: int) -> str:
        """
        Label index to name
        """
        return self._dataset_properties['label_indices'][str(label_index)]
    # end label_index_to_name

    # Label classes
    def label_classes(self, label_name: str) -> list:
        """
        Get label classes
        """
        label_index = self.label_name_to_index(label_name)
        return self._dataset_properties['labels'][label_index]['classes']
    # end label_classes

    # Label class names
    def label_class_names(self, label_name: str):
        """
        Get label class names
        """
        label_index = self.label_name_to_index(label_name)
        return self._dataset_properties['labels'][label_index]['class_names']
    # end label_class_names

    # Event type properties
    def event_type_properties(self, event_name: str) -> dict:
        """
        Event type properties
        """
        event_index = self.event_name_to_index(event_name)
        return self._dataset_properties['event_types'][event_index]
    # end event_type_properties

    # Event name to index
    def event_name_to_index(self, event_name: str) -> int:
        """
        Event name to index
        """
        return self._dataset_properties['event_type_names'][event_name]
    # end event_name_to_index

    # Event name from index
    def event_index_to_name(self, event_index: int) -> str:
        """
        Event index to name
        """
        return self._dataset_properties['event_type_indices'][str(event_index)]
    # end event_index_to_name

    # Column properties
    def column_properties(self, column_name: str) -> dict:
        """
        Column properties
        :param column_name: Column name
        :return: Dictionary of properties
        """
        column_id = self._dataset_properties['column_names']
        return self._dataset_properties['columns'][column_id]
    # end column_properties

    # Column index by name
    def column_name_to_index(self, column_name: str) -> int:
        """
        Column index by name
        """
        return self._dataset_properties['column_names'][column_name]
    # end column_index_from_name

    # Column name from index
    def column_index_to_name(self, column_index: int) -> str:
        """
        Column name from index
        """
        return self._dataset_properties['columns'][str(column_index)]['id']
    # end column_name_from_index

    # Get sample by index
    def get_sample(self, sample_index: int) -> dict:
        """
        Get sample by index
        """
        if self._sample_in_memory:
            return self._loaded_samples[sample_index]
        else:
            # Data file name
            data_file_path = os.path.join(self._root_directory, INFO_DATA_FILE_OUTPUT.format(sample_index))

            # Load tensor data
            return torch.load(open(data_file_path, 'rb'))
        # end if
    # end get_sample

    # Get sample metadata
    def get_sample_metadata(self, sample_index: int) -> dict:
        """
        Get sample metadata
        """
        # Metadata file
        metadata_file_path = os.path.join(self._root_directory, INFO_METADATA_FILE_OUTPUT.format(sample_index))

        # Load JSON
        return json.load(open(metadata_file_path, 'r'))
    # end get_sample_metadata

    # Get sample labels
    def get_sample_labels(self, sample_index: int) -> list:
        """
        Get sample labels
        """
        return self._dataset_properties['samples'][sample_index]['labels']
    # end get_sample_labels

    # Get sample class
    def get_sample_class(self, sample_index: int, label_index: int) -> int:
        """
        Get sample class
        """
        for sample_label in self.get_sample_labels(sample_index):
            if sample_label['id'] == label_index:
                return sample_label['class']
            # end if
        # end for
        raise Exception("Unknown label index: {}".format(label_index))
    # end get_sample_class

    # Get sample class tensor
    def get_sample_class_tensor(self, sample_index: int, time_tensor=False) -> torch.Tensor:
        """
        Get sample class tensor
        """
        if time_tensor:
            return self._create_class_time_tensor(
                self._dataset_properties['samples'][sample_index]['labels'],
                self.get_sample_length(sample_index)
            )
        else:
            n_labels = len(self.labels)
            class_array = np.zeros(n_labels)
            for label in self.labels:
                class_array[label['id']] = self.get_sample_class(sample_index, label['id'])
            # end for
            return torch.tensor(class_array)
        # end if
    # end get_sample_class_tensor

    # Get sample length
    def get_sample_length(self, sample_index: int) -> int:
        """
        Get sample length
        """
        return self._dataset_properties['samples'][sample_index]['length']
    # end get_sample_length

    # Get sample properties
    def get_sample_properties(self, sample_index: int) -> dict():
        """
        Get sample properties
        """
        return self._dataset_properties['samples'][sample_index]
    # end get_sample_properties

    # Get sample number of segments
    def get_sample_segment_count(self, sample_index: int) -> int:
        """
        Get sample number of segments
        """
        return self._dataset_properties['samples'][sample_index]['n_segments']
    # end get_sample_segment_count

    # Get sample segments
    def get_sample_segments(self, sample_index: int) -> list:
        """
        Get sample segments
        """
        return self._dataset_properties['samples'][sample_index]['segments']
    # end get_sample_segments

    # From segment label index to name
    def segment_label_index_to_name(self, segment_label_index: int) -> str:
        """
        From segment label index to name
        """
        return self._dataset_properties['segment_label_indices'][segment_label_index]
    # end segment_label_index_to_name

    # From segment label name to index
    def segment_label_name_to_index(self, segment_label_name: str) -> int:
        """
        From segment label name to index
        """
        return self._dataset_properties['segment_label_names'][segment_label_name]
    # end segment_label_name_to_index

    # Get metadata entry
    def get_metadata(self, item: str):
        """
        Get metadata entry
        """
        return self._dataset_properties['metadata'][item]
    # end get_metadata

    # endregion PUBLIC

    # region PRIVATE

    # Load JSON file
    def _load_properties_file(self, json_file: str) -> dict:
        """
        Load JSON file
        :param json_file: Path to JSON file
        """
        with open(json_file, 'r') as r:
            return json.load(r)
        # end with
    # end _load_json_file

    # Load samples in memory
    def _load_samples(self):
        """
        Load samples in memory
        """
        # Samples
        loaded_samples = list()

        # For each sample
        for sample_i in range(self.n_samples):
            loaded_samples.append(
                self.get_sample(sample_i)
            )
        # end for

        # In memory
        self._sample_in_memory = True

        return loaded_samples
    # end _loaded_samples

    # Create time tensor for labels
    def _create_class_time_tensor(self, sample_labels, sample_len) -> torch.Tensor:
        """
        Create time tensor for labels
        """
        # Np array
        class_array = np.zeros(shape=(sample_len, len(sample_labels)), dtype=np.long)

        # For each label
        for label in sample_labels:
            class_array[:, label['id']] = label['class']
        # end for

        return torch.tensor(class_array)
    # end _create_class_time_tensor

    # endregion PRIVATE

    # region OVERRIDE

    # Length of the dataset
    def __len__(self):
        """
        Length of the dataset
        """
        return self.n_samples
    # end __len__

    # Get an item
    def __getitem__(self, item: int) -> tuple:
        """
        Get an item
        """
        return (
            self.get_sample(item),
            self.get_sample_class_tensor(sample_index=item, time_tensor=True),
            self.get_sample_class_tensor(sample_index=item, time_tensor=False)
        )
    # end __getitem__

    # endregion OVERRIDE

    # region TO_OVERRIDE

    # Filter dataset
    def _filter_dataset(self, dataset_desc: dict) -> dict:
        """
        Filter dataset
        """
        return dataset_desc
    # end _filter_dataset

    # endregion TO_OVERRIDE

# end TimeseriesDataset

