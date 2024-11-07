import pytest
import matplotlib.pyplot as plt
import numpy as np

@pytest.mark.parametrize(
        "R_max, nR, nphi, nZ, R_0, I_c, nseg, tol",
        [[4,2,2,32,4,3,64, 0.05],]
        ##tolerance was chosen by comparing the formula for a circular loop and the formula for a polygon-shaped loop.
        ##the maximum value of the analytical solution was multiplied with the factor n*tan(pi/n)/pi
        ##the approximate difference was chosen as the tolerance
)

def test_biotsavart(R_max, nR, nphi, nZ, R_0, I_c, nseg, tol):
    """test the circular_current function to assert, whether the result using biotsavart_asdex is sufficiently
    close to the analytical solution with the example of a circular current loop

    Args:
        R_max (float): the extent of the computational grid, i.e., R ∈ [0,Rmax], Z ∈ [-Rmax,Rmax]
        nR (int), nphi (int), nZ (int): the number of grid points
        R_0 (float): Radius of the loop
        I_c (float): current flowing through the loop
        nseg (int): the number of segments in the discretisation of the loop
        tol (float): acceptable difference between the analytical result and the output of biotsavart_asdex
    """
    from reference_solutions import circular_current
    from reference_solutions import BZ_formula


    Z, BZ, BZ_analytic = circular_current(R_max, nR, nphi, nZ, R_0, I_c, nseg)
    plt.plot(Z, BZ, label='values from biotsavart.py')
    plt.plot(Z, BZ_analytic, label='analytical values')
    plt.legend()
    plt.xlabel('Z')
    plt.ylabel('BZ')
    plt.show()

    assert np.all([abs(BZ[i]-BZ_analytic[i]) < tol for i in range(nZ)])
