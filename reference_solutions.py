from bdivfree import get_A_field_modes, calc_B_field_modes
from biotsavart import make_field_file_from_coils
import numpy as np
import os
from plot_modes import read_field, read_field_hdf5, read_field_netcdf


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


def circular_current(
    R_max, nR, nphi, nZ, R_0, I_c, nseg, field_file, integrator, grid_iterator
):
    """Use biotsavart.py to calculate the magnetic field of a circular current loop

    Args:
        R_max (float): the extent of the computational grid, i.e., R ∈ [0,Rmax], Z ∈ [-Rmax,Rmax]
        nR (int), nphi (int), nZ (int): the number of grid points
        R_0 (float): Radius of the loop
        I_c (float): current flowing through the loop
        nseg (int): the number of segments in the discretisation of the loop
        field_file (str): File name of the magnetic field test calculation output.
        integrator (function, optional): Function to evaluate the Biot-Savart integral and calculate the magnetic field components.
        grid_iterator (function, optional): Function which iterates over the grid points onto which the magnetic field is calculated.
    Returns:
        Z (array[float], shape=(nZ, )): the array of equidistant Z coordinates of the computational grid
        BZ (array[float], shape=(nZ, )): the values of BZ(Z), as returned by the magnetic field calculation of biotsavart.py
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
        field_file,
        integrator,
        grid_iterator,
        1,
    )

    # Process output data
    if field_file.endswith(".h5") or field_file.endswith(".hdf5"):
        _, _, _, BZ = read_field_hdf5(field_file)
    elif field_file.endswith(".nc") or field_file.endswith(".cdf"):
        _, _, _, BZ = read_field_netcdf(field_file)
    else:
        _, _, _, BZ = read_field(field_file)
    BZ = BZ[0, 0, :]

    # Calculate the analytical solution
    Z = np.linspace(-R_max, R_max, nZ)
    BZ_analytic = [BZ_formula(z, R_0, I_c) for z in Z]

    # Remove test input/output files
    for filename in [
        "test_grid_file",
        "test_coil_file",
        "test_current_file",
        field_file,
    ]:
        os.remove(filename)

    return Z, BZ, BZ_analytic


def fourier_analysis(n_max, field_file):
    """Calculate the first 'n_max' modes of the magnetic field on the radial grid R via
    numpy.fft.fft used in get_A_field_modes and via a handwritten discrete fourier transformation.
    The input parameters and magnetic field components are read from field_file.

    Args:
        n_max (int): Highest mode number up to which the of the magnetic field is calculated.
        field_file (str): File name of the magnetic field calculation output.

    Returns:
        R (array[float], shape=(nR,)): Radial grid for which the magentic field modes are calculated.
        BnR_hand_written (array[complex float], shape=(n_max,nR)): R-component of the n'th mode of the magnetic field calculated
                                                                   by a handwritten fourier transformation method.
        BnR_fft (array[complex float], shape=(n_max,nR)): R-component of the n'th mode of the magnetic field calculated using
                                                          numpy.fft.fft (fast fourier transformation method).
    """
    # Get grid and magnetic field components from the calculation output field_file
    if field_file.endswith(".h5") or field_file.endswith(".hdf5"):
        grid, BR, Bphi, BZ = read_field_hdf5(field_file)
    elif field_file.endswith(".nc") or field_file.endswith(".cdf"):
        grid, BR, Bphi, BZ = read_field_netcdf(field_file)
    else:
        grid, BR, Bphi, BZ = read_field(field_file)

    # Calculate the first 'n_max' modes of the magnetic field on the radial grid ...

    # ... via the method numpy.fft.fft used in get_A_field_modes.
    A = get_A_field_modes(grid, BR, Bphi, BZ)
    BnR_fft = np.empty((n_max, grid.nR), dtype=complex)
    for k in range(n_max):
        # n=k+1 because range starts from 0 but n=1 is the first mode.
        # Fix Z to the middle of the axial grid
        BnR_fft[k], Bnphi, BnZ = calc_B_field_modes(
            grid.R, grid.Z[grid.nZ // 2], k + 1, A
        )

    # ... via a handwritten fourier transformation method.
    # The fourier transform implemented in get_A_field_modes is "forward".
    BnR_hand_written = np.empty((n_max, grid.nR), dtype=complex)
    for k in range(n_max):
        n = k + 1  # because range starts from 0 but n=1 is the first mode.
        # Calculate the fourier coefficients for the n'th mode.
        fourier_coefs = np.exp(-1j * n * np.linspace(0, 2 * np.pi, grid.nphi)) / (
            grid.nphi - 1
        )
        # Calculate the n'th mode of the radial component of the magnetic field.
        kZ = grid.nZ // 2  # Fix Z to the middle of the axial grid
        for kR in range(grid.nR):
            # phi goes from 0 to 2*pi therefore the end is excluded.
            BnR_hand_written[k, kR] = sum(BR[kR, :-1, kZ] * fourier_coefs[:-1])

    return grid.R, BnR_hand_written, BnR_fft
