from biotsavart import (
    calc_biotsavart,
    calc_biotsavart_vectorized,
    get_field_on_grid,
    get_field_on_grid_numba_parallel,
)
import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.mark.parametrize(
    "R_max, nR, nphi, nZ, R_0, I_c, nseg, field_file, integrator, grid_iterator",
    [
        [
            4,
            2,
            2,
            32,
            4,
            3,
            64,
            "field.h5",
            calc_biotsavart_vectorized,
            get_field_on_grid_numba_parallel,
        ],
    ],
)
def test_biotsavart(
    R_max, nR, nphi, nZ, R_0, I_c, nseg, field_file, integrator, grid_iterator
):
    """test the circular_current function to assert, whether the result using biotsavart.py is sufficiently
    close to the analytical solution with the example of a circular current loop

    Args:
        R_max (float): the extent of the computational grid, i.e., R ∈ [0,Rmax], Z ∈ [-Rmax,Rmax]
        nR (int), nphi (int), nZ (int): the number of grid points
        R_0 (float): Radius of the loop
        I_c (float): current flowing through the loop
        nseg (int): the number of segments in the discretisation of the loop
        field_file (str): File name of magnetic field test calculation output. File will be deleted after the test.
                          Extension determines which writer function is used.
        integrator (function): Function to evaluate the Biot-Savart integral and calculate the magnetic field components.
        grid_iterator (function, optional): Function which iterates over the grid points onto which the magnetic field is calculated.
        tol (float): acceptable difference between the analytical result and the output of biotsavart.py
    """
    from reference_solutions import circular_current

    Z, BZ, BZ_analytic = circular_current(
        R_max, nR, nphi, nZ, R_0, I_c, nseg, field_file, integrator, grid_iterator
    )
    plt.plot(Z, BZ, label="values from biotsavart.py")
    plt.plot(Z, BZ_analytic, label="analytical values")
    plt.legend()
    plt.xlabel("Z")
    plt.ylabel("BZ")
    plt.show()

    # Tolerance was chosen by comparing the formula for a circular loop and the formula for a polygon-shaped loop.
    # The maximum value of the analytical solution was multiplied with the factor nseg*tan(pi/nseg)/pi
    # and the machine epsilon was added to account for numerical errors.
    machine_epsilon = np.finfo(np.float64).eps
    tol = max(BZ_analytic) * (nseg * np.tan(np.pi / nseg) / np.pi - 1) + machine_epsilon

    assert np.all([abs(BZ[i] - BZ_analytic[i]) < tol for i in range(nZ)])
