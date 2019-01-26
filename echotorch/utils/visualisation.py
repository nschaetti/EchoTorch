# -*- coding: utf-8 -*-
#

# Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Display neurons activities on a 3D plot
def neurons_activities_3d(stats, neurons, title, timesteps=-1, start=0):
    """
    Display neurons activities on a 3D plot
    :param stats:
    :param neurons:
    :param title:
    :param timesteps:
    :param start:
    :return:
    """
    # Fig
    ax = plt.axes(projection='3d')

    # Two by two
    n_neurons = neurons.shape[0]
    stats = stats[:, neurons].view(-1, n_neurons // 3, 3)

    # Plot
    if timesteps == -1:
        time_length = stats.shape[0]
        ax.plot3D(stats[:, :, 0].view(time_length).numpy(), stats[:, :, 1].view(time_length).numpy(), stats[:, :, 2].view(time_length).numpy(), 'o')
    else:
        ax.plot3D(stats[start:start + timesteps, :, 0].numpy(), stats[start:start + timesteps, :, 1].numpy(), stats[start:start + timesteps, :, 2].numpy(), 'o', lw=0.5)
    # end if
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)
    plt.show()
    plt.close()
# end neurons_activities_3d


# Display neurons activities on a 2D plot
def neurons_activities_2d(stats, neurons, title, colors, timesteps=-1, start=0):
    """
    Display neurons activities on a 2D plot
    :param stats:
    :param neurons:
    :param title:
    :param timesteps:
    :param start:
    :return:
    """
    # Fig
    fig = plt.figure()
    ax = fig.gca()

    # Two by two
    n_neurons = neurons.shape[0]

    # For each plot
    for i, stat in enumerate(stats):
        # Stats
        stat = stat[:, neurons].view(-1, n_neurons // 2, 2)

        # Plot
        if timesteps == -1:
            ax.plot(stat[:, :, 0].numpy(), stat[:, :, 1].numpy(), colors[i])
        else:
            ax.plot(stat[start:start + timesteps, :, 0].numpy(), stat[start:start + timesteps, :, 1].numpy(), colors[i])
        # end if
    # end for
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_title(title)
    plt.show()
    plt.close()
# end neurons_activities_2d


# Display neurons activities
def neurons_activities_1d(stats, neurons, title, timesteps=-1, start=0):
    """
    Display neurons activities
    :param stats:
    :param neurons:
    :return:
    """
    # Fig
    fig = plt.figure()
    ax = fig.gca()

    if timesteps == -1:
        ax.plot(stats[:, neurons].numpy())
    else:
        ax.plot(stats[start:start + timesteps, neurons].numpy())
    # end if

    ax.set_xlabel("Timesteps")
    ax.set_title(title)
    plt.show()
    plt.close()
# end neurons_activities_1d


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
    plt.close()
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
    plt.close()
# end show_2d_timeseries


# Show 1D time series
def show_1d_timeseries(ts, title, start=0, timesteps=-1):
    """
    Show 1D time series
    :param ts:
    :param title:
    :return:
    """
    # Fig
    fig = plt.figure()
    ax = fig.gca()

    if timesteps == -1:
        ax.plot(ts[:, 0].numpy())
    else:
        ax.plot(ts[start:start+timesteps, 0].numpy())
    # end if
    ax.set_xlabel("X Axis")
    ax.set_title(title)
    plt.show()
    plt.close()
# end show_1d_timeseries
