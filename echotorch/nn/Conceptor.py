# -*- coding: utf-8 -*-
#
# File : echotorch/nn/ESN.py
# Description : An Echo State Network module.
# Date : 26th of January, 2018
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

"""
Created on 26 January 2018
@author: Nils Schaetti
"""

# Imports
import torch.sparse
import torch
from .RRCell import RRCell
import math
from echotorch.utils import generalized_squared_cosine
import math as m
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np


# Conceptor
class Conceptor(RRCell):
    """
    Conceptor
    """

    # Constructor
    def __init__(self, conceptor_dim, aperture=0.0, learning_algo='inv', name="", conceptor_matrix=None, dtype=torch.float32):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param output_dim: Reservoir size
        """
        super(Conceptor, self).__init__(conceptor_dim, conceptor_dim, ridge_param=aperture, feedbacks=False, with_bias=False, learning_algo=learning_algo, softmax_output=False, dtype=dtype)

        # Properties
        self.conceptor_dim = conceptor_dim
        self.aperture = aperture
        self.name = name
        self.n_samples = 0.0
        self.attenuation = 0.0

        # Set it as buffer
        self.register_buffer('R', Variable(torch.zeros(self.x_size, self.x_size, dtype=self.dtype), requires_grad=False))
        self.register_buffer('C', Variable(torch.zeros(1, conceptor_dim, dtype=self.dtype), requires_grad=False))

        # Set conceptor
        if conceptor_matrix is not None:
            self.C = conceptor_matrix
            self.train(False)
        # end if
    # end __init__

    ###############################################
    # PROPERTIES
    ###############################################

    # Compute quota
    @property
    def quota(self):
        """
        Compute quota
        :return:
        """
        # Conceptor matrix
        conceptor_matrix = self.get_C()

        # Compute sum of singular values devided by number of neurons
        return float(torch.sum(conceptor_matrix.mm(torch.eye(self.conceptor_dim, dtype=self.dtype))) / self.conceptor_dim)
    # end quota

    ###############################################
    # PUBLIC
    ###############################################

    # Plot 2D ellipse
    def plot(self, colorstring, linewidth=3, resolution=200, dim='2d'):
        """
        Plot 2D ellipse
        :return:
        """
        # 2D or 3D ?
        if dim == '2d':
            Conceptor.plot_ellipse_2D(self.get_C(), colorstring, linewidth, resolution)
        else:
            pass
        # end if
    # end plot

    # Clone
    def clone(self):
        """
        Clone
        :return:
        """
        return Conceptor(conceptor_dim=self.conceptor_dim, aperture=self.aperture, name=self.name, conceptor_matrix=self.C, dtype=self.dtype)
    # end clone

    # Show the conceptor matrix
    def show(self):
        """
        Show the conceptor matrix
        :return:
        """
        plt.imshow(self.C, cmap='Greys')
        plt.show()
    # end show

    # Change aperture
    def set_aperture(self, new_a):
        """
        Change aperture
        :param new_a:
        :return:
        """
        self.C = Conceptor.phi_function(self.C, new_a / self.aperture)
        self.aperture = new_a
    # end set_aperture

    # Multiply aperture
    def multiply_aperture(self, factor):
        """
        Multiply aperture
        :param factor:
        :return:
        """
        self.C = Conceptor.phi_function(self.C, factor)
        self.aperture *= factor
    # end multiply_aperture

    # Plot delta measure
    def plot_delta_measure(self, start, end, steps=50):
        """
        Plot delta measure
        :param start:
        :param end:
        :return:
        """
        # Gamma values
        gamma_values = torch.logspace(start=start, end=end, steps=steps)

        # Log10 of gamma values
        gamma_log_values = torch.log10(gamma_values)

        # Delta measures
        C_norms = torch.zeros(steps)
        delta_scores = torch.zeros(steps)

        # For each gamma measure
        for i, gamma in enumerate(gamma_values):
            delta_scores[i], C_norms[i] = self.delta_measure(float(gamma), epsilon=0.1)
        # end for

        # Plot
        plt.plot(gamma_log_values.numpy(), delta_scores.numpy())
        plt.plot(gamma_log_values.numpy(), C_norms.numpy())
        plt.show()
    # end plot_delta_measure

    # Compute Delta measure
    def delta_measure(self, gamma, epsilon=0.01):
        """
        Compute Delta measure
        :param gamma:
        :param epsilon:
        :return:
        """
        # Conceptor matrix for both sides
        A = Conceptor.phi_function(self.C, gamma - epsilon)
        B = Conceptor.phi_function(self.C, gamma + epsilon)

        # Gradient in Frobenius norm of matrix
        A_norm = math.pow(torch.norm(A, p=2), 2)
        B_norm = math.pow(torch.norm(B, p=2), 2)
        d_C_norm = B_norm - A_norm

        # Change in log(gamma)
        d_log_gamma = np.log(gamma + epsilon) - np.log(gamma - epsilon)
        """if d_C_norm / d_log_gamma > 50.0:
            print(A)
            print(B)
            print(torch.norm(A, p=2))
            print(torch.norm(B, p=2))
            print(d_C_norm)
            print(gamma)
            print(d_C_norm / d_log_gamma)
            exit()
        # end if"""
        return d_C_norm / d_log_gamma, d_C_norm
    # end delta_measure

    # Output matrix
    def get_C(self):
        """
        Output matrix
        :return:
        """
        return self.C
    # end get_w_out

    # Forward
    def forward(self, x, y=None):
        """
        Forward
        :param x: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        # Batch size
        batch_size = x.size(0)

        # Time length
        time_length = x.size(1)

        # Learning algo
        if self.training:
            for b in range(batch_size):
                Rj = x[b].t().mm(x[b]) / time_length
                self.R.data.add_(Rj.data)
                self.n_samples += 1.0
            # end for
            return x
        elif not self.training:
            # Outputs
            outputs = Variable(torch.zeros(batch_size, time_length, self.output_dim, dtype=self.dtype), requires_grad=False)
            outputs = outputs.cuda() if self.C.is_cuda else outputs

            # For each batch
            for b in range(batch_size):
                outputs[b] = torch.mm(x[b], self.C)
            # end for

            # Compute attenuation
            self.attenuation = torch.mean(torch.pow(torch.abs(x - outputs), 2)) / torch.mean(torch.pow(torch.abs(x), 2))

            return outputs
        # end if
    # end forward

    # Finish training
    def finalize(self):
        """
        Finalize training with LU factorization or Pseudo-inverse
        """
        # Average
        self.R = self.R / self.n_samples

        # SVF
        (U, S, V) = torch.svd(self.R)

        # Compute new singular values
        Snew = torch.mm(torch.diag(S), torch.inverse(torch.diag(S) + math.pow(self.aperture, -2) * torch.eye(self.input_dim, dtype=self.dtype)))

        # Apply new SVs to get the conceptor
        self.C.data = torch.mm(torch.mm(U, Snew), U.t()).data

        # Not in training mode anymore
        self.train(False)
    # end finalize

    # Set conceptor
    def set_conceptor(self, c):
        """
        Set conceptor
        :param c:
        :return:
        """
        # Set matrix
        self.w_out.data = c
    # end set_conceptor

    # Singular values
    def singular_values(self):
        """
        Singular values
        :return:
        """
        # Compute SVD
        (Ua, Sa, Va) = torch.svd(self.get_C())
        return Ua, torch.diag(Sa), Va
    # end singular_values

    # Some of singular values
    def get_quota(self):
        """
        Sum of singular values
        :return:
        """
        return float(torch.sum(self.singular_values()))
    # end get_quota

    ###############################################
    # STATIC
    ###############################################

    # Plot 2D ellipse
    @staticmethod
    def plot_ellipse_2D(A, colorstring, linewidth=3, resolution=200):
        """
        Plot 2D ellipse
        :return:
        """
        # Plot cross and circle
        plt.plot([-1, 1], [0, 0], '--', color='black', linewidth=1)
        plt.plot([0, 0], [-1, 1], '--', color='black', linewidth=1)
        plt.plot(
            np.cos(2.0 * math.pi * np.arange(200) / 200.0),
            np.sin(2.0 * math.pi * np.arange(200) / 200),
            '-',
            color='black',
            linewidth=1
        )

        # Compute the ellipse representing the correlation matrix
        circ_points = torch.from_numpy(np.array([
            np.cos(2.0 * math.pi * np.arange(0, resolution) / resolution),
            np.sin(2.0 * math.pi * np.arange(0, resolution) / resolution)
        ]))

        # Transform the circle
        E1 = torch.mm(A, circ_points)

        # SVD on A
        (U, S, Ut) = torch.svd(A)

        # Plot singular value 1
        plt.plot(
            S[0].item() * np.array([0., U[0, 0]]),
            S[0].item() * np.array([0., U[1, 0]]),
            linewidth=linewidth,
            color=colorstring
        )

        # Plot singular value 2
        plt.plot(
            S[1].item() * np.array([0., U[0, 1]]),
            S[1].item() * np.array([0., U[1, 1]]),
            linewidth=linewidth,
            color=colorstring
        )

        # Plot ellipse
        plt.plot(E1[0, :].numpy(), E1[1, :].numpy(), linewidth=linewidth, color=colorstring)
    # end plot_ellipse_2D

    # Multiply aperture matrix
    @staticmethod
    def phi_function(C, gamma):
        """
        Multiply aperture matrix
        :param c:
        :param gamma:
        :return:
        """
        # Conceptor matrix
        c = C.clone()
        conceptor_dim = c.shape[0]
        dtype = c.dtype

        # New tensor
        return c.mm(torch.inverse(c + m.pow(gamma, -2) * (torch.eye(conceptor_dim, dtype=dtype) - c)))
    # end phi_function

    # Morphing patterns
    @staticmethod
    def morphing(conceptor_list, mu):
        """
        Morphing pattern
        :param conceptor_list:
        :return:
        """
        # For each conceptors
        for i, c in enumerate(conceptor_list):
            if i == 0:
                M = c.mul(mu[i])
            else:
                M += c.mul(mu[i])
            # end if
        # end for
        return M
    # end for

    # Similarity between two conceptors
    @staticmethod
    def similarity(C1, C2):
        """
        Similarity between two conceptors
        :param C1:
        :param C2:
        :return:
        """
        # Compute singular values
        Ua, Sa, _ = torch.svd(C1.get_C())
        Ub, Sb, _ = torch.svd(C2.get_C())

        # Measure
        return generalized_squared_cosine(Sa, Ua, Sb, Ub)
    # end similarity

    ###############################################
    # OPERATORS
    ###############################################

    # Similarity with another conceptor
    def sim(self, cb, measure='gsc'):
        """
        Similarity with another conceptor
        :param cb:
        :return:
        """
        # Compute singular values
        Ua, Sa, _ = torch.svd(self.C)
        Ub, Sb, _ = torch.svd(cb.get_C())

        # Measure
        if measure == 'gsc':
            return generalized_squared_cosine(Sa, Ua, Sb, Ub)
        # end if
    # end sim

    # Positive evidence
    def E_plus(self, x):
        """
        Positive evidence
        :param x: states (x)
        :return:
        """
        return x.mm(self.w_out).mm(x.t())
    # end E_plus

    # Evidence against
    def E_neg(self, x, conceptor_list):
        """
        Evidence against
        :param x:
        :param conceptor_list:
        :return:
        """
        # For each conceptor in the list
        for i, c in enumerate(conceptor_list):
            if i == 0:
                new_c = c
            else:
                new_c = new_c.logical_or(c)
            # end if
        # end for

        # Take the not
        N = new_c.logical_not()

        return x.t().mm(N.w_out).mm(x)
    # end E_neg

    # Evidence
    def E(self, x, conceptor_list):
        """
        Evidence
        :param x:
        :param conceptor_list:
        :return:
        """
        return self.E_plus(x) + self.E_neg(x, conceptor_list)
    # end E

    # OR
    def logical_or(self, c):
        """
        Logical OR
        :param c:
        :return:
        """
        # Matrices
        C = self.C
        B = c.get_C()
        I = torch.eye(self.conceptor_dim, dtype=self.dtype)

        # Compute C1 \/ C2
        conceptor_matrix = torch.inverse(I + torch.inverse(C.mm(torch.inverse(I - C)) + B.mm(torch.inverse(I - B))))

        # Set conceptor
        new_c = Conceptor(
            conceptor_dim=self.conceptor_dim,
            conceptor_matrix=conceptor_matrix,
            name="({} OR {})".format(self.name, c.name),
            aperture=math.sqrt(math.pow(self.aperture, 2) + math.pow(c.aperture, 2)),
            dtype=self.dtype
        )

        return new_c
    # end logical_or

    # OR
    def __or__(self, other):
        """
        OR
        :param other:
        :return:
        """
        return self.logical_or(other)
    # end __or__

    # NOT
    def logical_not(self):
        """
        Logical NOT
        :param c:
        :return:
        """
        # Matrices
        C = self.C

        # Compute not C
        conceptor_matrix = torch.eye(self.conceptor_dim, dtype=self.dtype) - C

        # Set conceptor
        new_c = Conceptor(
            conceptor_dim=self.conceptor_dim,
            conceptor_matrix=conceptor_matrix,
            name="NOT {}".format(self.name),
            aperture=1.0 / self.aperture,
            dtype=self.dtype
        )

        return new_c
    # end logical_not

    # Not
    def __invert__(self):
        """
        NOT
        :return:
        """
        return self.logical_not()
    # end __invert__

    # AND
    def logical_and(self, C2):
        """
        Logical AND
        :param c:
        :return:
        """
        # Get conceptor matrix
        A = self.get_C()
        B = C2.get_C()

        # Dimension
        dim = A.shape[0]
        tol = 1e-14

        # SV on both conceptor
        (UC, SC, UtC) = torch.svd(A)
        (UB, SB, UtB) = torch.svd(B)

        # Get singular values
        dSC = SC
        dSB = SB

        # How many non-zero singular values
        numRankC = int(torch.sum(1.0 * (dSC > tol)))
        numRankB = int(torch.sum(1.0 * (dSB > tol)))

        if numRankC < dim and numRankB < dim:
            # Select zero singular vector
            UC0 = UC[:, numRankC:]
            UB0 = UB[:, numRankB:]

            # SVD on UC0 + UB0
            (W, Sigma, Wt) = torch.svd(torch.mm(UC0, UC0.t()) + torch.mm(UB0, UB0.t()))

            # Number of non-zero SV
            numRankSigma = int(torch.sum(1.0 * (Sigma > tol)))

            # Select zero singular vector
            Wgk = W[:, numRankSigma:]

            # C and B
            # Wgk * (Wgk^T * (C^-1 + B^-1 - I) * Wgk)^-1 * Wgk^T
            CandB = np.dot(
                np.dot(
                    Wgk,
                    torch.inverse(
                        np.dot(
                            np.dot(
                                Wgk.T,
                                (torch.pinverse(A, tol) + torch.pinverse(B, tol) - np.eye(dim))
                            ),
                            Wgk
                        )
                    )
                ),
                Wgk.T
            )
        else:
            # C and B
            # Wgk * (Wgk^T * (C^-1 + B^-1 - I) * Wgk)^-1 * Wgk^T
            CandB = torch.pinverse(A, tol) + torch.pinverse(B, tol) - torch.eye(dim, dtype=self.dtype)
        # end if

        # Set conceptor
        new_c = Conceptor(
            conceptor_dim=self.conceptor_dim,
            conceptor_matrix=CandB,
            name="({} AND {})".format(self.name, C2.name),
            aperture=math.pow(math.pow(self.aperture, -2) + math.pow(C2.aperture, -2), -0.5),
            dtype=self.dtype
        )

        return new_c
    # end logical_and

    # AND
    def __and__(self, other):
        """
        AND
        :param other:
        :return:
        """
        return self.logical_and(other)
    # end __and__

    # Multiply
    def mul(self, other):
        """
        Multiply
        :param other:
        :return:
        """
        # Multiply matrix
        if type(other) is Conceptor:
            new_c = self.get_C() * other.get_C()
        else:
            new_c = self.get_C() * other
        # end if

        # New conceptor
        return Conceptor(self.conceptor_dim, self.aperture, conceptor_matrix=new_c, dtype=self.dtype)
    # end mul

    # Multiply
    def __mul__(self, other):
        """
        Multiply
        :param other:
        :return:
        """
        # Multiply matrix
        if type(other) is Conceptor:
            new_c = self.get_C() * other.get_C()
        else:
            new_c = self.get_C() * other
        # end if

        # New conceptor
        return Conceptor(self.conceptor_dim, self.aperture, conceptor_matrix=new_c, dtype=self.dtype)
    # end __mul__

    # Multiply
    def __rmul__(self, other):
        """
        Multiply
        :param other:
        :return:
        """
        # Multiply matrix
        if type(other) is Conceptor:
            new_c = self.get_C() * other.get_C()
        else:
            new_c = self.get_C() * other
        # end if

        # New conceptor
        return Conceptor(self.conceptor_dim, self.aperture, conceptor_matrix=new_c, dtype=self.dtype)
    # end __mul__

    # Override *=
    def __imul__(self, other):
        """
        *=
        :param other:
        :return:
        """
        # Multiply matrix
        if type(other) is Conceptor:
            new_c = self.get_C() * other.get_C()
        else:
            new_c = self.get_C() * other
        # end if

        # New conceptor
        return Conceptor(self.conceptor_dim, self.aperture, conceptor_matrix=new_c, dtype=self.dtype)
    # end __imul__

    # Add
    def __add__(self, other):
        """
        Add
        :param other:
        :return:
        """
        # Add matrix
        if type(other) is Conceptor:
            new_c = self.get_C() + other.get_C()
        else:
            new_c = self.get_C() + other
        # end if

        # New conceptor
        return Conceptor(self.conceptor_dim, self.aperture, conceptor_matrix=new_c, dtype=self.dtype)
    # end __add__

    # Add
    def __radd__(self, other):
        """
        Add
        :param other:
        :return:
        """
        # Add matrix
        if type(other) is Conceptor:
            new_c = self.get_C() + other.get_C()
        else:
            new_c = self.get_C() + other
        # end if

        # New conceptor
        return Conceptor(self.conceptor_dim, self.aperture, conceptor_matrix=new_c, dtype=self.dtype)
    # end __radd__

    # +=
    def __iadd__(self, other):
        """
        +=
        :param other:
        :return:
        """
        # Add matrix
        if type(other) is Conceptor:
            new_c = self.get_C() + other.get_C()
        else:
            new_c = self.get_C() + other
        # end if

        # New conceptor
        return Conceptor(self.conceptor_dim, self.aperture, conceptor_matrix=new_c, dtype=self.dtype)
    # end __iadd__

    # Greater or equal
    def __ge__(self, other):
        """
        Greater or equal
        :param other:
        :return:
        """
        # Compute eigenvalues of a - b
        eig_v = torch.eig(other.get_C() - self.w_out, eigenvectors=False)
        return float(torch.max(eig_v)) >= 0.0
    # end __ge__

    # Greater
    def __gt__(self, other):
        """
        Greater
        :param other:
        :return:
        """
        # Compute eigenvalues of a - b
        eig_v = torch.eig(other.get_C() - self.w_out, eigenvectors=False)
        return float(torch.max(eig_v)) > 0.0
    # end __gt__

    # Less
    def __lt__(self, other):
        """
        Less than
        :param other:
        :return:
        """
        return not self >= other
    # end __lt__

    # Less or equal
    def __le__(self, other):
        """
        Less or equal
        :param other:
        :return:
        """
        return not self > other
    # end __le__

# end RRCell
