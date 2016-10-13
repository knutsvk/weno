"""Weighted Essential Non-Oscillatory (WENO) data reconstruction

Given a finite number of cells containing piecewise constant data, construct a
polynomial of arbitrary degree within each cell. The polynomial accurately
represents the underlying function on which the data is based, even in the
presence of discontinuities, and does so without introducing spurious
oscillations. Useful for e.g. numerical methods for hyperbolic PDEs.

Reference:
    "High order space-time adaptive ADER-WENO finite volume schemes for
    non-conservative hyperbolic systems"

    Michael Dumbser, Arturo Hidalgo, Olindo Zanotti
    Computer Methods in Applied Mechanics and Engineering
    Volume 268, 1 January 2014, Pages 359-387
    DOI: 10.1016/j.cma.2013.09.022
"""
import os.path
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import solve as linsol

from weno.funcs import nonlinear_weights, scale_func_arg, \
        weird_func
from weno.generate import basis_functions, oscillation_indicator_matrix, \
        stencil_coefficient_matrix
from weno.settings import PRECOMPUTE, ORDER

if PRECOMPUTE:
    basis = basis_functions(ORDER)
    osc_ind_mat = oscillation_indicator_matrix(ORDER)
    coef_mat = stencil_coefficient_matrix(ORDER)


def set_precompute(PreComp, Order=3):
    """Sets option to precompute basis polynomials and matrices.

    Parameters
    ----------
    PreComp : bool
        Precomputation is performed if equal to 'True'
    Order : int
        Order to which interpolation is correct

    Returns
    -------
    (void)
    """
    global PRECOMPUTE, ORDER
    PRECOMPUTE = PreComp
    ORDER = Order
    if PRECOMPUTE:
        global basis, osc_ind_mat, coef_mat
        basis = basis_functions(ORDER)
        osc_ind_mat = oscillation_indicator_matrix(ORDER)
        coef_mat = stencil_coefficient_matrix(ORDER)


def reconstruct(x, y, N=ORDER-1):
    """Returns WENO reconstruction in each cell.

    Given positions of cell centers, piecewise constant and degree of
    interpolating polynomials, returns a WENO reconstruction for each cell.

    Parameters
    ----------
    x : ndarray
        1-D ndarray containing positions of cell centers
    y : ndarray
        1-D ndarray containing piecewise constant data in each cell
    N : int
        Degree of interpolating polynomials

    Returns
    -------
    recon : dict
        Dictionary of WENO reconstructions represented as numpy.poly1d
        instances
    """
    cell_amount = len(x)
    dx = x[1] - x[0]

    if PRECOMPUTE:
        global basis
        global osc_ind_mat
        global coef_mat
        assert ORDER == N + 1
    else:
        basis = basis_functions(N + 1)
        osc_ind_mat = oscillation_indicator_matrix(N + 1)
        coef_mat = stencil_coefficient_matrix(N + 1)

    # Extend domain to both sides, copy function value in last cell out to
    # ghost cells.  This allows for reconstructions in the whole domain. 
    x = np.concatenate((x[0] - dx * np.arange(N + 1, 1, -1), x, x[-1] + dx *
        np.arange(1, N + 1)))
    y = np.concatenate((y[0] * np.ones(N), y, y[-1] * np.ones(N)))

    recon = {}
    for i in range(N, N + cell_amount):
        print(i)

        # Set amount of stencils  based on whether N is odd or even
        # Get piecewise constant values of y within cells
        if N % 2:
            num_stencils = 4
            y_cells = [
                y[i - int(np.floor(N/2)):i + int(np.ceil(N/2)) + 1],
                y[i - int(np.ceil(N/2)):i + int(np.floor(N/2)) + 1],
                y[i - N:i + 1],
                y[i:i + N + 1]]
        else:
            num_stencils = 3
            y_cells = [
                y[i - int(N/2):i + int(N/2) + 1],
                y[i - N:i + 1],
                y[i:i + N + 1]]

        # Generate weights for each stencil by solving linear systems
        stenc_weights = linsol(coef_mat, y_cells)

        # Compute oscillation indicator for each stencil
        osc_ind = np.array([sw.dot(osc_ind_mat).dot(sw) for sw in
                            stenc_weights])

        # Get final weight for each basis function
        weights = nonlinear_weights(stenc_weights, osc_ind, num_stencils)

        # Sum together contributions from each basis function, scale argument
        recon[i - N] = basis[0] * weights[0]
        for j in range(1, N + 1):
            recon[i - N] = np.polyadd(recon[i - N], basis[j] * weights[j])
        recon[i - N] = scale_func_arg(recon[i - N], x[i], dx)
    return recon


def decon_recon(domain=[0, 12], func=weird_func, poly_deg=ORDER-1,
                num_cells=30, do_plot=True):
    """Deconstruction and reconstruction of data.

    Demonstration of deconstruction of analytical function to piecewise
    constant data and subsequent reconstruction using the WENO procedure.

    Parameters
    ----------
    domain : double[2]
        Start and end points of domain
    func : function handle
        Analytical function to be deconstructed and reconstructed again
    poly_deg : int
        Degree of interpolating polynomials
    num_cells : int
        Number of cells to decompose domain into
    do_plot : bool
        If true, generate plot showing  analytical function and reconstructed
        polynomials

    Returns
    -------
    (void)
    """
    # Generate array of cell center positions x_i = domain[0] + (i + 1/2) * dx
    x, dx = np.linspace(domain[0], domain[1], num_cells, endpoint=False,
                        retstep=True)
    x += dx / 2

    # Deconstruct analytical function to piecewise constant data within cells
    piecewise_constant = func(x)

    # Perform WENO reconstruction
    reconstruction = reconstruct(x, piecewise_constant, poly_deg)

    if do_plot:
        plt.clf()

        # Plot analytical function with large amount of points
        x_dense = np.linspace(domain[0], domain[1], 100 * num_cells)
        plt.plot(x_dense, func(x_dense), 'r-')
        plt.xlim(domain)

        # Plot the reconstructed polynomial function within each cell
        for i in range(0, num_cells):
            x_cell = np.linspace(x[i] - 0.5 * dx, x[i] + 0.5 * dx)
            plt.plot(x_cell, reconstruction[i](x_cell), 'b-')
        plt.show()
