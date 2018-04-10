# -*- coding: utf-8 -*-
#

# Imports
import mdp
import numpy as np
import matplotlib.pyplot as plt

# Init. random
np.random.seed(0)

# Parameters
n = 10000
p2 = np.pi * 2
t = np.linspace(0, 1, n, endpoint=0)
dforce = np.sin(p2*5*t) + np.sin(p2*11*t) + np.sin(p2*13*t)


def logistic_map(x, r):
    return r*x*(1-x)
# end logistic_map

# Series
series = np.zeros((n, 1), 'd')
series[0] = 0.6

# Create series
for i in range(1, n):
    series[i] = logistic_map(series[i-1], 3.6+0.13*dforce[i])
# end for

# MDP flow
flow = (mdp.nodes.EtaComputerNode() +
        mdp.nodes.TimeFramesNode(10) +
        mdp.nodes.PolynomialExpansionNode(3) +
        mdp.nodes.SFA2Node(output_dim=1) +
        mdp.nodes.EtaComputerNode())

# Train
flow.train(series)

# Slow
slow = flow(series)

resc_dforce = (dforce - np.mean(dforce, 0)) / np.std(dforce, 0)

print(u"{}".format(mdp.utils.cov2(resc_dforce[:-9], slow)))
print(u"Eta value (time serie) : {}".format(flow[0].get_eta(t=10000)))
print(u"Eta value (slow feature) : {}".format(flow[-1].get_eta(t=9996)))
