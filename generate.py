"""Functions to generate WENO basis functions and matrices
"""
import os.path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import lagrange
from .settings import PRECOMPUTE, ORDER


def basis_functions(N):
    """Return nodal basis functions.

    Given an integer, 'N', generate the nodal basis functions which correspond
    to the N Legendre nodes, shifted to the interval (0, 1).

    Parameters
    ----------
    N : int
        Order of reconstructed polynomials

    Returns
    -------
    polys : dict
        Dictionary of basis functions represented as numpy.poly1d instances
    """
    legendre_nodes, gauss_weights = leggauss(N)
    legendre_nodes = (legendre_nodes + 1) / 2

    polys = {}
    for i in range(N):
        kronecker_delta = np.zeros(N)
        kronecker_delta[i] = 1
        polys[i] = lagrange(legendre_nodes, kronecker_delta)

    return polys


def stencil_coefficient_matrix(N):
    """Return coefficient matrices for stencils.

    In order to find the weights in the linear combination of basis functions
    within each cell of a stencil, we need to solve the linear system A w = u,
    where w are the weights and u are the piecewise constant data. A is the
    stencil coefficient matrix.

    Parameters
    ----------
    N : int
        Order of reconstructed polynomials

    Returns
    -------
    mat : ndarray
        Matrices represented as :numpy.array: of shape (num_stencils, N, N)
        containing coefficients for each stencil, cell and basis function.
    """
    basis = basis_functions(N)

    # Use stencil of 3 cells if the reconstruction order is odd, and 4 if even
    if N % 2:
        num_stencils = 3
    else:
        num_stencils = 4

    mat = np.zeros((num_stencils, N, N))

    # Loop over cells in stencil
    for c in range(N):
        # Loop over basis functions
        for f, func in basis.items():
            antideriv = np.polyint(func)

            if num_stencils == 3:
                # Central stencil
                mat[0, c, f] = antideriv(c - int((N - 1) / 2) + 1) \
                        - antideriv(c - int((N - 1) / 2))
                # Left stencil
                mat[1, c, f] = antideriv(c - N + 2) - antideriv(c - N + 1)
                # Right stencil
                mat[2, c, f] = antideriv(c + 1) - antideriv(c)
            else:
                # Central left stencil
                mat[0, c, f] = antideriv(c - int(np.floor((N - 1) / 2)) + 1) \
                    - antideriv(c - int(np.floor((N - 1) / 2)))
                # Central right stencil
                mat[1, c, f] = antideriv(c - int(np.ceil((N - 1) / 2)) + 1) \
                    - antideriv(c - int(np.ceil((N - 1) / 2)))
                # Left stencil
                mat[2, c, f] = antideriv(c - N + 2) - antideriv(c - N + 1)
                # Right stencil
                mat[3, c, f] = antideriv(c + 1) - antideriv(c)
    return mat


def oscillation_indicator_matrix(N):
    """Return oscillation indicator matrix.

    The matrix is independent of data and used to compute the oscillation
    indicator for a stencil once its weights have been computed.

    Parameters
    ----------
    N : int
        Order of reconstructed polynomials

    Returns
    -------
    mat : ndarray
        Oscillation indicator matrix as :numpy.array: of shape (N, N).
    """
    basis = basis_functions(N)

    mat = np.zeros((N, N))
    for deriv_order in range(1, N):
        for i in range(N):
            for j in range(N):
                mat[i, j] += np.polyint(np.polymul(
                        np.polyder(basis[i], deriv_order),
                        np.polyder(basis[j], deriv_order)))(1.0)
    return mat
