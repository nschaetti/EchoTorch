# -*- coding: utf-8 -*-
#

# Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Show 3D time series
def show_3d_timeseries(ts, title):
    """
    Show 3D timeseries
    :param axis:
    :param title:
    :return:
    """
    # Fig
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(ts[:, 0].numpy(), ts[:, 1].numpy(), ts[:, 2].numpy(), lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)
    plt.show()
# end show_3d_timeseries


# Show 2D time series
def show_2d_timeseries(ts, title):
    """
    Show 2D timeseries
    :param ts:
    :param title:
    :return:
    """
    # Fig
    fig = plt.figure()
    ax = fig.gca()

    ax.plot(ts[:, 0].numpy(), ts[:, 1].numpy(), lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_title(title)
    plt.show()
# end show_2d_timeseries


# Show 1D time series
def show_1d_timeseries(ts, title):
    """
    Show 1D time series
    :param ts:
    :param title:
    :return:
    """
    # Fig
    fig = plt.figure()
    ax = fig.gca()

    ax.plot(ts[:, 0].numpy())
    ax.set_xlabel("X Axis")
    ax.set_title(title)
    plt.show()
# end show_1d_timeseries
