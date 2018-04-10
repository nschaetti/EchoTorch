# -*- coding: utf-8 -*-
#

# Imports
import mdp
import numpy as np

# Init. random
np.random.seed(0)

# Parameters
p2 = np.pi * 2
t = np.linspace(0, 1, 10000, endpoint=0)
dforce = np.sin(p2*5*t) + np.sin(p2*11*t) + np.sin(p2*13*t)


def logistic_map(x, r):
    return r*x*(1-x)
# end logistic_map

# Series
series = np.zeros((10000, 1), 'd')
series[0] = 0.6

# Create series
for i in range(1, 10000):
    series[i] = logistic_map(series[i-1], 3.6+0.13*dforce[i])
# end for

print(series)
