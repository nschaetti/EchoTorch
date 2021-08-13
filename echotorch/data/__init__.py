# -*- coding: utf-8 -*-
#
# File : echotorch/datasets/__init__.py
# Description : Dataset subpackages init file
# Date : 3th of March, 2021
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

# Import functions
from .functional.chaotic import henon
from .functional.random_processes import random_walk

# Imports datasets
from .CopyTaskDataset import CopyTaskDataset
from .DatasetComposer import DatasetComposer
from .DelayDataset import DelayDataset
from .DiscreteMarkovChainDataset import DiscreteMarkovChainDataset
from .EchoDataset import EchoDataset
from .FromCSVDataset import FromCSVDataset
from .HenonAttractor import HenonAttractor
from .ImageToTimeseries import ImageToTimeseries
from .LambdaDataset import LambdaDataset
from .LatchTaskDataset import LatchTaskDataset
from .LogisticMapDataset import LogisticMapDataset
from .LorenzAttractor import LorenzAttractor
from .MackeyGlass2DDataset import MackeyGlass2DDataset
from .MackeyGlassDataset import MackeyGlassDataset
from .MarkovChainDataset import MarkovChainDataset
from .MemTestDataset import MemTestDataset
from .MixedSinesDataset import MixedSinesDataset
from .NARMADataset import NARMADataset
from .RosslerAttractor import RosslerAttractor
from .SinusoidalTimeseries import SinusoidalTimeseries
from .PeriodicSignalDataset import PeriodicSignalDataset
from .RandomSymbolDataset import RandomSymbolDataset
from .RepeatTaskDataset import RepeatTaskDataset
from .TimeseriesBatchSequencesDataset import TimeseriesBatchSequencesDataset
from .TimeseriesDataset import TimeseriesDataset
from .TransformDataset import TransformDataset
from .TripletBatching import TripletBatching

__all__ = [
   # Functionals
   'henon', 'random_walk',
   # Datasets
   'CopyTaskDataset', 'DatasetComposer', 'DiscreteMarkovChainDataset', 'FromCSVDataset', 'HenonAttractor',
   'LambdaDataset', 'LatchTaskDataset', 'LogisticMapDataset', 'LorenzAttractor', 'MackeyGlassDataset', 'MemTestDataset',
   'NARMADataset', 'RosslerAttractor', 'SinusoidalTimeseries', 'PeriodicSignalDataset', 'RandomSymbolDataset',
   'ImageToTimeseries', 'MarkovChainDataset', 'MixedSinesDataset', 'RepeatTaskDataset',
   'TimeseriesBatchSequencesDataset', 'TransformDataset', 'TripletBatching', 'DelayDataset', 'EchoDataset',
   'MackeyGlass2DDataset'
]
