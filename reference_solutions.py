from bdivfree import get_A_field_modes, calc_B_field_modes
from biotsavart import make_field_file_from_coils
import numpy as np
import os
from plot_modes import read_field


def BZ_formula(z, R, I):
    """Analytical formula for the magnetic field of a circular current loop along its axis

    Args:
        z (float): position along the axis
        R (float): Radius of the loop
        I (float): current flowing through the loop

    Returns:
        Magnetic field Bz (float) at position z along the axis of the loop
    """
    return I * R**2 * 2 * np.pi / (R**2 + z**2) ** (3 / 2)


def circular_current(R_max, nR, nphi, nZ, R_0, I_c, nseg, integrator, grid_iterator):
    """Use biotsavart.py to calculate the magnetic field of a circular current loop

    Args:
        R_max (float): the extent of the computational grid, i.e., R ∈ [0,Rmax], Z ∈ [-Rmax,Rmax]
        nR (int), nphi (int), nZ (int): the number of grid points
        R_0 (float): Radius of the loop
        I_c (float): current flowing through the loop
        nseg (int): the number of segments in the discretisation of the loop
        integrator (function, optional): Function to evaluate the Biot-Savart integral and calculate the magnetic field components.
        grid_iterator (function, optional): Function which iterates over the grid points onto which the magnetic field is calculated.

    Returns:
        Z (array[float], shape=(nZ, )): the array of equidistant Z coordinates of the computational grid
        BZ (array[float], shape=(nZ, )): the values of BZ(Z), as returned by biotsavart_asdex
        BZ_analytical (array[float], shape=(nZ, )): the values of BZ(Z), as returned by the analytical formula
    """
    # Create input files for test
    with open("test_grid_file", "w") as coil_file:
        coil_file.write(f"{nR} {nphi} {nZ} \n")
        coil_file.write(f"{0} {R_max} \n")
        coil_file.write(f"{-R_max} {R_max} \n")

    with open("test_coil_file", "w") as coil_file:
        coil_file.write(f"{nseg + 1} \n")

        dphi = 2 * np.pi / nseg
        X = R_0 * np.cos(np.arange(nseg) * dphi)
        X = np.append(X, R_0)  # Close the loop
        Y = R_0 * np.sin(np.arange(nseg) * dphi)
        Y = np.append(Y, 0)  # Close the loop
        Z = np.zeros_like(X)
        has_current = np.ones_like(X)
        has_current[-1] = 0  # The last point does not have current
        coil_number = np.ones_like(X)
        np.savetxt(
            coil_file,
            np.column_stack((X, Y, Z, has_current, coil_number)),
        )

    with open("test_current_file", "w") as current_file:
        current_file.write(f"{I_c} \n")

    # Run the calculation
    make_field_file_from_coils(
        "test_grid_file",
        "test_coil_file",
        "test_current_file",
        "test_field_file",
        integrator,
        grid_iterator,
        1,
    )

    # Process output data
    _, _, _, BZ = read_field("test_field_file")
    BZ = BZ[0, 0, :]

    # Calculate the analytical solution
    Z = np.linspace(-R_max, R_max, nZ)
    BZ_analytic = [BZ_formula(z, R_0, I_c) for z in Z]

    # Remove test input/output files
    for filename in [
        "test_grid_file",
        "test_coil_file",
        "test_current_file",
        "test_field_file",
    ]:
        os.remove(filename)

    return Z, BZ, BZ_analytic


def fourier_analysis(n_max):
    """Calculate the first 'n_max' modes of the magnetic field on the radial grid R via handwritten discrete fourier transform
    using field_c and via numpy.fft.fft. The input parameters and magnetic field components are read from field_file.

    Args:
        n_max (int): Highest mode number up to which the of the magnetic field is calculated.

    Returns:
        R (array[float], shape=(nR,)): Radial grid for which the magentic field modes are calculated.
        BnR (array[complex float], shape=(n_max,nR)): R-component of the n'th mode of the magnetic field calculated
                                                      by field_c using a handwritten fourier transform.
        BnR_fft (array[complex float], shape=(n_max,nR)): R-component of the n'th mode of the magnetic field calculated
                                                      by numpy.fft.fft using fast fourier transform.
    """
    # Get grid and magnetic field components from the calculation output field_file
    g, BR, Bphi, BZ = read_field("field_file")

    # Calculate the first 'n_max' modes of the magnetic field on the radial grid ...

    # ... via the handwritten fourier transform in field_c
    A = get_A_field_modes(g, BR, Bphi, BZ)
    BnR = np.empty((n_max, g.nR), dtype=complex)
    for k in range(n_max):
        # n=k+1 because range starts from 0 but n=1 is the first mode.
        # Fix Z to the middle of the axial grid
        BnR[k], Bnphi, BnZ = calc_B_field_modes(g.R, g.Z[g.nZ // 2], k + 1, A)

    # ... via the method numpy.fft.fft.
    BnR_fft = np.empty((n_max, g.nR))
    # The fourier transform implemented in field_c is "forward".
    # phi goes from 0 to 2*pi therefore the end is excluded.
    # numpy stores the components such that the first mode is on index 1 and the higher order modes follow on the next indices.
    BnR_fft = np.fft.fft(BR[:, :-1, g.nZ // 2], norm="forward")[:, 1 : n_max + 1]
    BnR_fft = BnR_fft.T  # Change to the same shape as BnR.
    return g.R, BnR, BnR_fft
