# -*- coding: utf-8 -*-
#
# File : papers/schaetti2016/transforms/Concat.py
# Description : Transform images to a concatenation of multiple transformations.
# Date : 11th of November, 2019
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
import math
import torch


# Compute conceptor
def compute_conceptor(X, aperture):
    """
    Compute conceptor
    :param X: Reservoir states (T x Nx)
    :param aperture: Aperture
    """
    x_length = X.size(0)
    x_dim = X.size(1)
    Rx = torch.mm(X.t(), X) / x_length
    Cx = torch.mm(torch.inverse(Rx + math.pow(aperture, -2) * torch.eye(x_dim)), Rx)
    URx, SRx, _ = torch.svd(Rx)
    UCx, SCx, _ = torch.svd(Cx)
    return Rx, Cx, URx, SRx, UCx, SCx
# end compute_conceptor


# Generalised cosine similarity
def gcsim(Ua, Sa, Ub, Sb):
    """
    Generalised cosine similarity
    :param Ua: First singular vectors
    :param Sa: First singular values
    :param Ub: Second singular vectors
    :param Sb: Second singular vectors
    :return Generalised cosine similarity
    """
    Sa = torch.diag(Sa)
    Sb = torch.diag(Sb)
    Va = torch.sqrt(Sa).mm(Ua.t())
    Vb = Ub.mm(torch.sqrt(Sb))
    Vab = Va.mm(Vb)
    num = torch.pow(torch.norm(Vab), 2)
    den = torch.norm(torch.diag(Sa), p=2) * torch.norm(torch.diag(Sb), p=2)
    return num / den
# end gcsim


# Conceptor Similarity Triplet Loss
def CSTLoss(A, P, N, aperture, margin, sv_requires_grad=False):
    """
    Conceptor Similarity Triplet Loss
    :param A: Anchor sample
    :param P: Positive sample
    :param N: Negative sample
    :param aperture: Aperture
    :param margin: Margin
    """
    # Compute conceptor matrix for A
    RA, CA, URA, SRA, UCA, SCA = compute_conceptor(A, aperture)
    RP, CP, URP, SRP, UCP, SCP = compute_conceptor(P, aperture)
    RN, CN, URN, SRN, UCN, SCN = compute_conceptor(N, aperture)

    # Gradient on singular values
    if not sv_requires_grad:
        SCA = SCA.detach()
        SCP = SCP.detach()
        SCN = SCN.detach()
    # end if

    return gcsim(UCA, SCA, UCN, SCN) - gcsim(UCA, SCA, UCP, SCP) + margin
# end CSTLoss
