# -*- coding: utf-8 -*-
#
# File : echotorch/series.py
# Description : Utility functions to generate timeseries
# Date : 25th of January, 2021
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
from typing import Union
import torch
import torchvision
import echotorch.datasets as etds
import echotorch.utils.evaluation as etev
import echotorch.transforms.images as etim
import echotorch.transforms.targets as etta


# Generate Copy Task series
def copytask(
        size: tuple, length_min: int, length_max: int, n_inputs: int, return_db: bool = False, dtype=None
) -> Union[etds.EchoDataset, torch.Tensor]:
    """
    Generate a dataset or series for the Copy task
    @param size: How many series to generate
    @param length_min: Minimum length
    @param length_max: Maximum length
    @param n_inputs: Number of inputs
    @param return_db: Return the datset (True) or series (False)
    @param dtype: Data type of the output series
    @return: An EchoDataset or a tensor of series
    """
    # The dataset
    dataset = etds.CopyTaskDataset(
        n_samples=size,
        length_min=length_min,
        length_max=length_max,
        n_inputs=n_inputs,
        dtype=dtype
    )

    if return_db:
        return dataset
    else:
        return dataset.data
    # end if
# end copytask


# Compose a dataset
def compose(
        datasets: list
) -> etds.EchoDataset:
    """
    Compose a dataset from a list of datasets
    @param datasets: A list of datasets
    @return: A new EchoDataset composed of the dataset in the given list
    """
    return etds.DatasetComposer(
        datasets=datasets
    )
# end compose


# Create cross validation dataset
def cross_eval(
        root_dataset, k=10, dev_ratio=0, shuffle=False, train_size=1.0, fold=0, mode='train', sample_indices=None,
        return_multiple_dataset=False
) -> Union[etds.EchoDataset, torch.Tensor]:
    """
    Create a cross validation dataset from a root dataset
    @param root_dataset: Root dataset (EchoDataset)
    @param k: Number of folds
    @param dev_ratio: Ratio of the dev set
    @param shuffle:
    @param train_size:
    @param fold:
    @param mode:
    @param sample_indices:
    @param return_multiple_dataset:
    @return:
    """
    if not return_multiple_dataset:
        return etev.CrossValidationWithDev(
            root_dataset=root_dataset,
            k=k,
            mode=mode,
            samples_indices=sample_indices,
            fold=fold,
            train_size=train_size,
            dev_ratio=dev_ratio,
            shuffle=shuffle
        )
    else:
        cv10_datasets = dict()
        for dataset_type in ['train', 'dev', 'test']:
            cv10_datasets[dataset_type] = etev.CrossValidationWithDev(
                root_dataset=root_dataset,
                k=k,
                dev_ratio=dev_ratio,
                shuffle=shuffle,
                train_size=train_size,
                fold=fold,
                mode=dataset_type,
                samples_indices=sample_indices
            )
        # end for
        return cv10_datasets
    # end if
# cross_eval


# Load Time series from a CSV file
def csv_file(
        csv_file: str, delimiter: str, quotechar: str, columns: list, return_db: bool = False, dtype=None
) -> Union[etds.EchoDataset, torch.Tensor]:
    """
    Load Timeseries from a CSV file
    :param csv_file:
    :param delimiter:
    :param quotechar:
    :param columns:
    :param return_db:
    :param dtype:
    """
    if return_db:
        return etds.FromCSVDataset(
            csv_file=csv_file,
            columns=columns,
            delimiter=delimiter,
            quotechar=quotechar,
            dtype=dtype
        )
    else:
        return etds.FromCSVDataset.generate(
            csv_file=csv_file,
            delimiter=delimiter,
            quotechar=quotechar,
            columns=columns,
            dtype=dtype
        )
    # end if
# end csv_file


# Delay dataset
def delaytask(
        root_dataset: etds.EchoDataset, delay: int, data_index: int = 0, keep_indices: bool = None
) -> etds.EchoDataset:
    """
    Delay dataset
    """
    return etds.DelayDataset(
        root_dataset=root_dataset,
        n_delays=delay,
        data_index=data_index,
        keep_indices=keep_indices
    )
# end delaytask


# Generate Discrete Markov Chain dataset
def discrete_markov_chain(
        size, length, n_states, probability_matrix, start_state=0, return_db=False, dtype=None
) -> Union[etds.EchoDataset, torch.Tensor]:
    """
    Generate series of Discrete Markov Chain directly or through a dataset
    @param size:
    @param length:
    @param n_states:
    @param probability_matrix:
    @param start_state:
    @param return_db:
    @param dtype:
    @return:
    """
    if return_db:
        return etds.DiscreteMarkovChainDataset(
            n_samples=size,
            sample_length=length,
            probability_matrix=probability_matrix
        )
    else:
        samples = list()
        for sample_i in range(size):
            samples.append(etds.DiscreteMarkovChainDataset.generate(
                length=length,
                n_states=n_states,
                probability_matrix=probability_matrix,
                start_state=start_state,
                dtype=dtype
            ))
        # end for
        return samples
    # end if
# end discrete_markov_chain


# Henon attractor
def henon(
        size: int,
        length: int,
        xy: int,
        a: int,
        b: int,
        washout: int = 0,
        normalize: bool = False,
        return_db: bool = False,
        dtype=None
) -> Union[etds.EchoDataset, torch.Tensor]:
    """Generate a series with the Hénon map dynamical system.

    The Hénon-Pomean attractor is a dynamical system which exhibit chaotic behavior. Each point :math:`(x_n, y_n)` in
    the plane is mapped to the new point

    .. math::
        \sum_{i=1}^{\\infty} x_{i}

    :param size: How many samples to generate
    :type size: int
    :param length: Length of samples (time)
    :type length: int
    :param xy: Parameter
    :type xy: int
    :param a: Parameter
    :type a: int
    :param b: Parameter
    :type b: int
    :param washout: Time steps to remove at the beginning of samples
    :type washout: int
    :param normalize: Normalize samples
    :type normalize: bool
    :param return_db: Return the database object
    :type return_db: bool
    :param dtype: Tensor data type
    :type dtype: ``torch.dtype``

    Examples::
        >>> echotorch.henon(1, 100, 1, 2, 3)
        timetensor([...])
    """
    if return_db:
        return etds.HenonAttractor(
            sample_len=length,
            n_samples=size,
            xy=xy,
            a=a,
            b=b,
            washout=washout,
            normalize=normalize
        )
    else:
        return etds.HenonAttractor.generate(
            n_samples=size,
            sample_len=length,
            xy=xy,
            a=a,
            b=b,
            washout=washout,
            normalize=normalize,
            dtype=dtype
        )
    # end if
# end henon


# From images to time series
def images(
        image_dataset: etds.EchoDataset,
        n_images: int,
        transpose: bool
) -> Union[etds.EchoDataset, torch.Tensor]:
    return etds.ImageToTimeseries(
        image_dataset=image_dataset,
        n_images=n_images,
        transpose=transpose
    )
# end images


# Create series from a function
def lambda_dataset(
        size: int, length: int, func: callable, start: int = 0, dtype=None
) -> Union[etds.EchoDataset, torch.Tensor]:
    """
    Create series from a function
    @param size:
    @param length:
    @param func:
    @param start:
    @param dtype:
    @return:
    """
    return etds.LambdaDataset(
        sample_len=length,
        n_samples=size,
        func=func,
        start=start,
        dtype=dtype
    )
# end lambda_dataset


# Latch task dataset
def latch(
        size: int, length_min: int, length_max: int, n_pics: int, dtype=None
) -> Union[etds.EchoDataset, torch.Tensor]:
    """
    Latch task dataset
    @param size:
    @param length_min:
    @param length_max:
    @param n_pics:
    @param dtype:
    @return:
    """
    return etds.LatchTaskDataset(
        n_samples=size,
        length_min=length_min,
        length_max=length_max,
        n_pics=n_pics,
        dtype=dtype
    )
# end latch


# Dataset from the logistic map
def logistic_map(
        size: int, length: int, alpha: float = 5, beta: float = 11, gamma: float = 13, c: float = 3.6, b: float = 0.13,
        seed: int = None
) -> Union[etds.EchoDataset, torch.Tensor]:
    """
    Dataset from the logistic map
    @param size:
    @param length:
    @param alpha:
    @param beta:
    @param gamma:
    @param c:
    @param b:
    @param seed:
    @return:
    """
    return etds.LogisticMapDataset(
        sample_len=length,
        n_samples=size,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        c=c,
        b=b,
        seed=seed
    )
# end logistic_map


# Mackey Glass time series
def mackey_glass(
        size, length, tau=17, return_db=False, dtype=None
) -> Union[etds.EchoDataset, torch.Tensor]:
    """
    Mackey Glass timeseries
    """
    if return_db:
        return etds.MackeyGlassDataset(
            sample_len=length,
            n_samples=size,
            tau=tau
        )
    else:
        samples = list()
        for sample_i in range(size):
            return etds.MackeyGlassDataset.generate(

            )
        # end for
    # end if
# end mackey_glass


# Mackey Glass time series
def mackey_glass_2d(
        size, length, subsample_rate, tau=17, normalize=False, seed=None, return_db=False, dtype=None
) -> Union[etds.EchoDataset, torch.Tensor]:
    """
    Mackey Glass timeseries
    """
    if return_db:
        return etds.MackeyGlass2DDataset(
            sample_len=length,
            n_samples=size,
            tau=tau,
            subsample_rate=subsample_rate,
            normalize=normalize
        )
    else:
        pass
    # end if
# end mackey_glass


# Markov Chain Dataset
def markov_chain(
        size: int, length: int, datasets: list, states_length: int, morphing_length: int,
        probability_matrix: torch.Tensor, random_start: int = 0, *args, **kwargs
) -> Union[etds.EchoDataset, torch.Tensor]:
    return etds.MarkovChainDataset(
        datasets=datasets,
        states_length=states_length,
        morphing_length=morphing_length,
        n_samples=size,
        sample_length=length,
        probability_matrix=probability_matrix,
        random_start=random_start,
        *args,
        **kwargs
    )
# end markov_chain


# MemTest dataset
def memtest(
        size: int, length: int, n_delays: int = 10, seed: int = None
) -> Union[etds.EchoDataset, torch.Tensor]:
    """
    MemTest dataset
    @param size:
    @param length:
    @param n_delays:
    @param seed:
    @return:
    """
    return etds.MemTestDataset(
        sample_len=length,
        n_samples=size,
        n_delays=n_delays,
        seed=seed
    )
# end memtest


# NARMA
def narma(
        size: int, length: int, order: int = 10, return_db: bool = False, dtype=None
) -> Union[etds.EchoDataset, torch.Tensor]:
    """
    Generate NARMA-x series or dataset
    @param size:
    @param length:
    @param order:
    @param return_db:
    @param dtype:
    @return:
    """
    if return_db:
        return etds.NARMADataset(
            sample_len=length,
            n_samples=size,
            system_order=order
        )
    else:
        return etds.NARMADataset.generate(
            sample_len=length,
            n_samples=length,
            system_order=order,
            dtype=dtype
        )
    # end if
# end narma


# NARMA-10
def narma10(
        size: int, length: int, return_db: bool = False, dtype=None
) -> Union[etds.EchoDataset, torch.Tensor]:
    """
    NARMA-10
    """
    if return_db:
        return etds.NARMADataset(
            sample_len=length,
            n_samples=size,
            system_order=10
        )
    else:
        return etds.NARMADataset.generate(
            sample_len=length,
            n_samples=length,
            system_order=10,
            dtype=dtype
        )
    # end if
# end narma10


# NARMA-30
def narma30(
        size: int, length: int, return_db: bool = False, dtype=None
) -> Union[etds.EchoDataset, torch.Tensor]:
    """
    NARMA-30
    """
    if return_db:
        return etds.NARMADataset(
            sample_len=length,
            n_samples=size,
            system_order=30
        )
    else:
        return etds.NARMADataset.generate(
            sample_len=length,
            n_samples=length,
            system_order=30,
            dtype=dtype
        )
    # end if
# end narma30


# Segment series
def segment_series(
        root_dataset: etds.EchoDataset, window_size: int, data_indices: list, stride: int, remove_indices: list,
        time_axis: int = 0, dataset_in_memory: bool = False, *args, **kwargs
) -> etds.EchoDataset:
    """
    Segment the series in a dataset with a sliding window
    @param root_dataset:
    @param window_size:
    @param data_indices:
    @param stride:
    @param remove_indices:
    @param time_axis:
    @param dataset_in_memory:
    @param args:
    @param kwargs:
    @return:
    """
    return etds.TimeseriesBatchSequencesDataset(
        root_dataset=root_dataset,
        window_size=window_size,
        data_indices=data_indices,
        stride=stride,
        remove_indices=remove_indices,
        time_axis=time_axis,
        dataset_in_memory=dataset_in_memory,
        *args,
        **kwargs
    )
# end segment_series


# MNIST series and dataset
def mnist(
        image_size: int, degrees: list, root: str = ".", download: bool = True, block_size: int = 100, return_db=False
) -> Union[etds.EchoDataset, torch.Tensor]:
    """
    Load (and download) the MNIST dataset
    @param image_size: Final image size (after crop and resize)
    @param degrees: List of rotation degrees to apply
    @param root: Root directory for the dataset (if downloaded)
    @param download: Download the dataset if not present ?
    @param block_size: The number of image per block
    @param return_db: True to return the dataset, False otherwise
    @return: Dataset or Tensor
    """
    # Concat rotation and crop
    transforms = [etim.CropResize(size=image_size)]

    # Add each composition
    for degree in degrees:
        transforms.append(
            torchvision.transforms.Compose([
                etim.Rotate(degree=degree),
                etim.CropResize(size=image_size)
            ])
        )
    # end for

    # Create the dataset (train)
    train_dataset = etds.ImageToTimeseries(
        torchvision.datasets.MNIST(
            root=root,
            train=True,
            download=download,
            transform=torchvision.transforms.Compose([
                etim.Concat(transforms, sequential=True),
                torchvision.transforms.ToTensor()
            ]),
            target_transform=etta.ToOneHot(class_size=10)
        ),
        n_images=block_size
    )

    # Create the dataset (test)
    test_dataset = etds.ImageToTimeseries(
        torchvision.datasets.MNIST(
            root=root,
            train=False,
            download=download,
            transform=torchvision.transforms.Compose([
                etim.Concat(transforms, sequential=True),
                torchvision.transforms.ToTensor()
            ]),
            target_transform=etta.ToOneHot(class_size=10)
        ),
        n_images=block_size
    )

    # If db or tensor
    if return_db:
        return train_dataset, test_dataset
    else:
        return train_dataset.generate(), test_dataset.generate()
    # end if
# end mnist
