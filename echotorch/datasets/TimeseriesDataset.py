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


# A dataset to load time series from a directory with meta-data
class TimeseriesDataset(EchoDataset):
    """
    A dataset to load time series from a directory with meta-data
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(self, root_directory, global_transform=None, transforms=None, global_label_transform=None,
                 label_transforms=None, timestep=1.0, selected_columns=None, segment_label_to_return=None,
                 in_memory=False, return_segments=True, return_events=True, dtype=torch.float64):
        """
        Constructor
        :param root_directory: Base root directory
        :param global_transform: An EchoTorch transformer for the whole time series
        :param transforms: EchoTorch transformers for each segment labels
        :parma global_label_transform:
        :param timestep: The time step of time series (default: 1 second)
        :param selected_columns: Names of the columns to return in the tensor
        :param segment_label_to_return: Segment label to return ('all' for no selection)
        :param in_memory: Keep data in memory
        :param return_segments: Return segments as a tensor
        :param return_events: Return events as a tensor
        :param dtype: Data type
        """
        # Properties
        self._root_directory = root_directory
        self._timestep = timestep
        self._root_json_file = os.path.join(self._root_directory, PROPERTIES_FILE)
        self._in_memory = False
        self._segment_label_to_return = segment_label_to_return
        self._return_segments = return_segments
        self._return_events = return_events
        self._dtype = dtype

        # Global transformer
        self._global_transform = global_transform
        self._global_label_transform = global_label_transform

        # Transformers
        if transforms is None:
            self._transforms = dict()
        else:
            self._transforms = transforms
        # end if

        # Label transformers
        if label_transforms is None:
            self._label_transforms = dict()
        else:
            self._label_transforms = label_transforms
        # end if

        # Load JSON file
        self._dataset_properties = self._filter_dataset(
            self._load_properties_file(self._root_json_file)
        )

        # Selected columns
        if selected_columns is None:
            self._selected_columns = self.columns
        else:
            self._selected_columns = selected_columns
        # end if

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

    # Global transform (GET)
    @property
    def global_transform(self) -> Transformer:
        """
        Global transform (GET)
        :return: A Transformer object
        """
        return self._global_transform
    # end global_transform

    # Global transform (SET)
    @global_transform.setter
    def global_transform(self, value: Transformer) -> None:
        """
        Global transform (SET)
        :param value: New transformer
        """
        self._global_transform = value
    # end global_transform

    # Global label transform (GET)
    @property
    def global_label_transform(self) -> Transformer:
        """
        Global transform (GET)
        :return: A Transformer object
        """
        return self._global_label_transform

    # end global_label_transform

    # Global transform (SET)
    @global_label_transform.setter
    def global_label_transform(self, value: Transformer) -> None:
        """
        Global transform (SET)
        :param value: New transformer
        """
        self._global_label_transform = value
    # end global_label_transform

    # Transforms (GET)
    @property
    def transforms(self) -> dict:
        """
        Transform
        :return: Transformer
        """
        return self._transforms
    # end transforms

    # Transform (SET)
    @transforms.setter
    def transforms(self, value: dict):
        """
        Transformer (SET)
        :param value: New transformers as a dict
        """
        self._transforms = value
    # end transforms

    # Label transforms (GET)
    @property
    def label_transforms(self) -> dict:
        """
        Transform
        :return: Transformer
        """
        return self._label_transforms
    # end label_transforms

    # Label transform (SET)
    @label_transforms.setter
    def label_transforms(self, value: dict):
        """
        Transformer (SET)
        :param value: New transformers as a dict
        """
        self._label_transforms = value
    # end label_transforms

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

    # Segment label names
    @property
    def segment_label_names(self) -> list:
        """
        Segment label names
        :return: Segment label names as a list
        """
        return self._dataset_properties['segment_label_names']
    # end segment_label_names

    # Segment label indices
    @property
    def segment_label_indices(self) -> list:
        """
        Segment label indices
        :return: Segment label indices as a list
        """
        return self._dataset_properties['segment_label_indices']
    # end segment_label_indices

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

    # Class name to index
    def class_name_to_index(self, label_index: int, class_name: str) -> int:
        """
        Class name to index
        """

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
            return torch.tensor(class_array, dtype=self._dtype)
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

    # Get sample events
    def get_sample_events(self, sample_index: int) -> list:
        """
        Get sample events
        """
        return self._dataset_properties['samples'][sample_index]['events']
    # end get_sample_events

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

    # Get transformer for a gait
    def get_transform(self, segment_label_name: str) -> Transformer:
        """
        Get transformer for a gait
        :param segment_label_name: Segment label name
        :return: The transformer
        """
        return self._transforms[segment_label_name]
    # end get_transform

    # Set transformer for a gait
    def set_transform(self, segment_label_name: str, value: Transformer):
        """
        Set transformer for a gait
        :param segment_label_name: Segment label name
        :param value: New transformer
        """
        self._transforms[segment_label_name] = value
    # end set_transform

    # Get segment label stats
    def get_segment_label_stats(self, segment_label_name: str) -> dict:
        """
        Get segment label stats
        """
        segment_label_index = self.segment_label_name_to_index(segment_label_name)
        return self._dataset_properties["segment_labels"][segment_label_index]["stats"]
    # end get_segment_label_stats

    # endregion PUBLIC

    # region PRIVATE

    # Apply transformers
    def _apply_transformers(self, global_transform: Transformer, transforms: dict, data_tensor: torch.Tensor,
                            sample_segments: list) -> torch.Tensor:
        """
        Apply transformers
        :param data_tensor: Sample data tensors as a dict
        :param sample_segments: Sample segments as a dict
        :return: Transformed data tensors as a dict
        """
        # Apply global transformers
        if global_transform is not None:
            data_tensor = global_transform(data_tensor)
        # end if

        # For each segment
        for segment in sample_segments:
            # Get segment info
            segment_start = segment['start']
            segment_end = segment['end']
            segment_label = segment['label']

            # There is a transformer for this?
            if segment_label in transforms.keys() and transforms[segment_label] is not None:
                data_tensor[segment_start:segment_end] = transforms[segment_label](
                    data_tensor[segment_start:segment_end]
                )
            # end if
        # end for

        return data_tensor
    # end _apply_transformers

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

        return torch.tensor(class_array).long()
    # end _create_class_time_tensor

    # Create a tensor from the dictionary
    def _create_input_tensor(self, timeseries_dict: dict, sample_length: int) -> torch.Tensor:
        """
        Create a tensor from the dictionary
        :param timeseries_dict: Dictionary of tensors for each columns
        :param sample_length: Length of the sample
        :return: The tensor with all columns
        """
        # Create an empty float tensor
        timeseries_input = torch.empty(sample_length, len(self._selected_columns), dtype=self._dtype)

        # For each data column
        for col_i, col_name in enumerate(self._selected_columns):
            if col_name in timeseries_dict.keys():
                timeseries_input[:, col_i] = timeseries_dict[col_name]
            # end if
        # end for

        return timeseries_input
    # end _create_input_tensor

    # Create a tensor for segments
    def _create_segments_tensor(self, sample_segments: list, segment_label_name: str) -> torch.Tensor:
        """
        Create a tensor for segments
        :param sample_segments: Sample segments
        :param segment_label_name:
        :return: Segments position and end as a tensor
        """
        # List of segments
        gait_segments_list = list()

        # Name to index
        segment_label_index = None
        if segment_label_name is not None:
            segment_label_index = self.segment_label_name_to_index(segment_label_name)
        # end if

        # Last starting position
        last_start = 0

        # For each segment
        for segment in sample_segments:
            if segment_label_name is None:
                gait_segments_list.append([segment['start'], segment['end'], segment['label']])
            elif segment['label'] == segment_label_index:
                gait_segments_list.append([last_start, last_start + segment['length'], segment['label']])
                last_start += segment['length']
            # end for
        # end for

        # return gait_segments_tensor
        return torch.LongTensor(gait_segments_list)
    # end _create_gait_segments_tensor

    # Time t in a segment of gait type?
    def _pos_in_segment_label(self, segments: list, event: dict, segment_label_name: str):
        """
        Time t in a segment of gait type?
        :param segments:
        :param segment_label_name:
        :return:
        """
        # Name to index
        segment_label_index = self.segment_label_name_to_index(segment_label_name)

        last_start = 0
        for segment in segments:
            if segment['label'] == segment_label_index:
                if segment['start'] <= event['start'] <= segment['end'] and segment['start'] <= event['end'] <= \
                        segment['end']:
                    return {
                        'start': last_start + (event['start'] - segment['start']),
                        'end': last_start + (event['end'] - segment['start'])
                    }
                # end if
                last_start += segment['length']
            # end if
        # end for
        return None

    # end _pos_in_gait_type_segment

    # Create a tensor for events (jumps)
    def _create_events_tensor(self, sample_events: list, sample_segments: list, segment_label_name: str) -> torch.Tensor:
        """
        Create a  tensor for events (jumps)
        :param sample_events:
        :return:
        """
        # Create an empty tensor for events data
        events_list = list()

        # For each events
        for event in sample_events:
            if segment_label_name is None:
                events_list.append([event['start'], event['end'], event['type']])
            else:
                found_event = self._pos_in_segment_label(sample_segments, event, segment_label_name)
                # If in the segment
                if found_event is not None:
                    events_list.append([found_event['start'], found_event['end'], event['type']])
                # end if
            # end if
        # end for

        return torch.LongTensor(events_list)
    # end _create_events_tensor

    # Filter a timeseries as tensor for gait type
    def _filter_ts_segment_label_name(self, timeseries_input: torch.Tensor, sample_segments: list,
                                      segment_label_name: str) -> torch.Tensor:
        """
        Filter a timeseries as tensor for gait type
        :param timeseries_input:
        :param sample_segments:
        :param segment_label_name:
        :return:
        """
        if segment_label_name is not None:
            # List of indices in the temporal axis
            indices_list = []

            # For each segment
            for segment in sample_segments:
                if segment['label'] == self.segment_label_name_to_index(segment_label_name):
                    indices_list += list(range(segment['start'], segment['end']))
                # end if
            # end for

            return torch.index_select(timeseries_input, dim=0, index=torch.LongTensor(indices_list))
        else:
            return timeseries_input
        # end if
    # end _filter_ts_gait_type

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
        # Return list
        return_list = []

        # Get sample from Timeseries dataset
        timeseries_dict = self.get_sample(item)
        class_time_tensor = self.get_sample_class_tensor(sample_index=item, time_tensor=True)
        class_tensor = self.get_sample_class_tensor(sample_index=item, time_tensor=False)

        # Create a tensor from the dictionary
        timeseries_input = self._create_input_tensor(timeseries_dict, self.get_sample_length(item))

        # Apply transforms to input timeseries
        timeseries_input = self._apply_transformers(
            self._global_transform,
            self._transforms,
            timeseries_input,
            self.get_sample_segments(item)
        )

        # Filter input data for targeted segment label
        timeseries_input = self._filter_ts_segment_label_name(
            timeseries_input,
            self.get_sample_segments(item),
            self._segment_label_to_return
        )

        # Filter time-related ground truth for targeted segment label
        class_time_tensor = self._filter_ts_segment_label_name(
            class_time_tensor,
            self.get_sample_segments(item),
            self._segment_label_to_return
        )

        # Apply label transforms to the class label timeseries
        class_time_tensor = self._apply_transformers(
            self._global_label_transform,
            self._label_transforms,
            class_time_tensor,
            self.get_sample_segments(item)
        )

        # Add to returns
        return_list += [timeseries_input, class_time_tensor, class_tensor]

        # Create the tensor for segments (if needed)
        if self._return_segments:
            gait_segments_tensor = self._create_segments_tensor(
                self.get_sample_segments(item),
                self._segment_label_to_return
            )
            return_list.append(gait_segments_tensor)
        # end if

        # Create the tensor for events (jumps) (if needed)
        if self._return_events:
            events_tensor = self._create_events_tensor(
                self.get_sample_events(item),
                self.get_sample_segments(item),
                self._segment_label_to_return
            )
            return_list.append(events_tensor)
        # end if

        return return_list
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

