# -*- coding: utf-8 -*-
#
# File : echotorch/nn/reservoir/__init__.py
# Description : nn/reservoir init file.
# Date : 29th of October, 2019
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
from .Conceptor import Conceptor
from .ConceptorNet import ConceptorNet
from .ConceptorSet import ConceptorSet
from .IncConceptorNet import IncConceptorNet
from .IncForgSPESNCell import IncForgSPESNCell
from .IncSPESN import IncSPESN
from .IncSPESNCell import IncSPESNCell
from .SPESN import SPESN
from .SPESNCell import SPESNCell

# All
__all__ = [
    'Conceptor', 'ConceptorNet', 'ConceptorSet', 'IncForgSPESNCell', 'IncConceptorNet', 'IncSPESN', 'IncSPESNCell',
    'SPESN', 'SPESNCell'
]
