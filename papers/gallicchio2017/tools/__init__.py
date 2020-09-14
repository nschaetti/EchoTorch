# -*- coding: utf-8 -*-
#
# File : papers/gallicchio2017/tools/__init__.py
# Description : Reproduction of the paper "Deep Reservoir Computing : A Critical Experiemental Analysis"
# (Gallicchio 2017)
# Date : 10th of September, 2020
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
# Copyright Nils Schaetti <nils.schaetti@unine.ch>, <nils.schaetti@unige.ch>

# Imports
from .evaluate_perturbations import evaluate_perturbations
from .tools import euclidian_distances, perturbation_effect, ranking_of_layers, kendalls_tau, spearmans_rule, timescales_separation

# ALL
__all__ = ['evaluate_perturbations', 'euclidian_distances', 'perturbation_effect', 'ranking_of_layers', 'kendalls_tau',
           'spearmans_rule', 'timescales_separation']
