"""Helper functions for WENO reconstruction
"""
import numpy as np
from .settings import EPS, R, LAMBDA_SIDE, LAMBDA_CENTRE


def nonlinear_weights(sten_w, osc_ind, num_stencils):
    """Return weights for basis functions.

    Given weights and oscillation indicators for each stencil, computes the
    weights used in assembling the final reconstruction as a nonlinear sum of
    basis functions.

    Parameters
    ----------
    sten_w : ndarray
        1-D ndarray containing weights for each stencil
    osc_ind : ndarray
        1-D ndarray containing oscillation indicators for each stencil
    num_stencils: int
        Number of stencils employed

    Returns
    -------
    w : ndarray
        1-D ndarray containing weights for basis functions
    """
    # Set up order of stencils
    if num_stencils == 4:
        lam = [LAMBDA_CENTRE, LAMBDA_CENTRE, LAMBDA_SIDE, LAMBDA_SIDE]
    else:
        lam = [LAMBDA_CENTRE, LAMBDA_SIDE, LAMBDA_SIDE]

    # Compute weights for each stencil weight
    omega = lam / (osc_ind + EPS) ** R
    omega /= omega.sum()

    return omega.dot(sten_w)


def scale_func_arg(func, x_i, dx):
    """Returns a function copy with scaled input variable.

    Basis functions are given in terms of the scaled space variable
    chi = (x-x_i)/dx + 1/2, but the reconstruction should be in terms of x.
    Given a function of x, it is returned with scaled input.

    Parameters
    ----------
    func : function handle
        Function that takes chi as argument
    x_i : double
        Location of left cell interface
    dx: double
        Cell width

    Returns
    -------
    f : function handle
        Function that takes x as argument
    """
    def f(x):
        return func((x - x_i) / dx + 1 / 2)
    return f


def weird_func(x):
    """Sine-wave and a couple of discontinuities to put WENO to the test"""
    y = np.zeros(len(x))
    y[x < 4] = 1
    y += np.sin(x)
    y[x > 6] = -3 + 0.3 * x[x > 6]
    return y
