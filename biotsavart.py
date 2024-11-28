# Dominik Haberz 11916636
# Clara Rinner 01137166
#
import numpy as np
from grid import grid
import argparse


class coils:
    def __init__(self, X, Y, Z, has_current, coil_number, n_nodes):
        """Object to store the coil data.

        Attributes:
            X (array[float], shape=(n_nodes, )): x-coodinate of the coil points
            Y (array[float], shape=(n_nodes, )): y-coodinate of the coil points
            Z (array[float], shape=(n_nodes, )): z-coodinate of the coil points
            has_current (array[bool], shape=(n_nodes, )): 0 if last point in a coil, otherwise 1
            coil_number (array[bool], shape=(n_nodes, )): Number of the coil, which each point belongs to.
            n_nodes (int): Total number of coil points.
        """
        self.X = X
        self.Y = Y
        self.Z = Z
        self.has_current = has_current
        self.coil_number = coil_number
        self.n_nodes = n_nodes


def calc_biotsavart(grid_coordinates, coils, currents):
    """
    Calculate the magnetic field components by evaluating the Biot-Savart integral for the
    given coil geometry and currents.

    Args:
        grid_coordinates (array[float], shape=(3)): cylindrical coordinates of the grid point
        coils (coils object): coil data
        currents (List[float], length=n_coils): Currents of each coil. n_coils is the total number of coils
    Returns:
        BR (float): Radial component fo the magnetic field
        Bphi (float): Toroidal component of the magnetic field
        BZ (float): Axial component of the magnetic field
    """

    R_grid = grid_coordinates[0]
    phi_grid = grid_coordinates[1]

    cosf = np.cos(phi_grid)
    sinf = np.sin(phi_grid)
    Y_grid = R_grid * sinf
    X_grid = R_grid * cosf
    Z_grid = grid_coordinates[2]
    grid_point = [X_grid, Y_grid, Z_grid]

    B = [0, 0, 0]

    coil_point = [coils.X[0], coils.Y[0], coils.Z[0]]

    # Distance between grid point and first coil point: |r-r'[0]|
    R1_vector = np.subtract(grid_point, coil_point)
    R1 = np.linalg.norm(R1_vector)

    for K in range(1, coils.n_nodes):  # loops through the remaining coil points
        coil_point_previous = coil_point
        coil_point = [coils.X[K], coils.Y[K], coils.Z[K]]

        # Difference between the current coil point and the previous one.
        L = np.subtract(coil_point, coil_point_previous)
        # Difference between grid point and current coil point: | r-r'[k]
        R2_vector = np.subtract(grid_point, coil_point)
        # Scalar product of L and r-r'[k]
        scalar_product = np.dot(L, R2_vector)
        # Distance between grid point and current coil point: |r-r'[k]|
        R2 = np.linalg.norm(R2_vector)

        if coils.has_current[K - 1] != 0.0:  # If not first point of a coil
            factor1 = 1.0 / (R2 * (R1 + R2) + scalar_product)
            factor2 = (
                -(R1 + R2) * factor1 / R1 / R2 * currents[coils.coil_number[K] - 1]
            )
            cross_product = np.cross(R2_vector, L)  # r-r'[k] x L

            B = np.add(np.dot(cross_product, factor2), B)

        R1 = R2

    BR = B[0] * cosf + B[1] * sinf
    Bphi = B[1] * cosf - B[0] * sinf
    BZ = B[2]

    return BR, Bphi, BZ


def read_coils(coil_file):
    """Read the input data stored in coil_file and return a coils object.

    Args:
        coil_file (str): Input file containing the coil geometry.
    Returns:
        coils object with the parameters read from the coil_file.
    """
    data = np.loadtxt(coil_file, skiprows=1)
    n_nodes = len(data)
    X = data[:, 0]
    Y = data[:, 1]
    Z = data[:, 2]
    has_current = data[:, 3]
    coil_number = [int(d) for d in data[:, 4]]

    last_nodes_idxs = np.where(np.diff(coil_number) != 0)[0]

    # Raise error if the current factor for the last node of each coil is not 0.
    if np.any(has_current[last_nodes_idxs] != 0.0):
        raise Exception(str(has_current[n_nodes - 1]) + "=cbc~=0, stop")

    return coils(X, Y, Z, has_current, coil_number, n_nodes)


def read_currents(current_file):
    """Return the input data stored in current_file.

    Args:
        current_file (str): Input file containing the currents of each coil.
    Returns:
        currents (array[float], shape=(n_coils, )): Currents of each coil. n_coils is the total number of coils.

    """
    with open(current_file, "r") as f:
        currents = [float(data) for data in f.readline().split()]
    return currents


def read_grid(grid_file, field_periodicity=1):
    """Read the input data stored in grid_file and return a grid object.

    Args:
        grid_file (str): Input file containing the parameters for a discretized grid.
        field_periodicity (int): Periodicity of the field in phi direction used for Tokamaks. Defaults to 1.

    Returns:
        grid object with the parameters read from the grid_file.
    """
    with open(grid_file, "r") as f:
        nR, nphi, nZ = [int(data) for data in f.readline().split()]
        R_min, R_max = [float(data) for data in f.readline().split()]
        Z_min, Z_max = [float(data) for data in f.readline().split()]
    phi_min = 0
    phi_max = 2 * np.pi / field_periodicity

    return grid(nR, nphi, nZ, R_min, R_max, phi_min, phi_max, Z_min, Z_max)


def get_field_on_grid(grid, coils, currents):
    """Calculate the magnetic field components for the discretized grid .

    Args:
        grid (grid object): Contains the parameters for the discretized grid.
        coils (coils object): coil data
        currents (array[float], shape=(n_coils, )): Currents of each coil. n_coils is the total number of coils.

    Returns:
        BR (array[float], shape=(n_points, )): Radial component fo the magnetic field. n_points is the total number of grid points.
        Bphi (array[float], shape=(n_points, )): Toroidal component of the magnetic field. n_points is the total number of grid points.
        BZ (array[float], shape=(n_points, )): Axial component of the magnetic field. n_points is the total number of grid points.
    """
    n_points = grid.nR * grid.nphi * grid.nZ
    BR = np.empty((n_points))
    Bphi = np.empty((n_points))
    BZ = np.empty((n_points))
    i = 0
    for r in grid.R:
        for p in grid.phi:
            for z in grid.Z:
                x = [r, p, z]
                BR[i], Bphi[i], BZ[i] = calc_biotsavart(x, coils, currents)
                i += 1
    return BR, Bphi, BZ


def write_field_to_file(field_file, grid, BR, Bphi, BZ, field_periodicity):
    """Write the calculation parameters (grid, field_periodicity) and the magnetic field components to the output file.

    Args:
        field_file (str): Output file into which the magnetic field components and calculation parameters are written to.
        grid (grid object): Contains the parameters for the discretized grid.
        BR (array[float], shape=(n_points, )): Radial component fo the magnetic field. n_points is the total number of grid points.
        Bphi (array[float], shape=(n_points, )): Toroidal component of the magnetic field. n_points is the total number of grid points.
        BZ (array[float], shape=(n_points, )): Axial component of the magnetic field. n_points is the total number of grid points.
        field_periodicity (int): Periodicity of the field in phi direction used for Tokamaks.
    """

    with open(field_file, "w") as f:
        # Write the input parameters for the magnetic field calculation.
        f.write(f"{grid.nR} {grid.nphi} {grid.nZ} {field_periodicity}\n")
        f.write(f"{grid.R_min} {grid.R_max}\n")
        f.write(f"{grid.phi_min} {grid.phi_max}\n")
        f.write(f"{grid.Z_min} {grid.Z_max}\n")
        # Write the the magnetic field components.
        np.savetxt(f, np.column_stack((BR, Bphi, BZ)))


def make_field_file_from_coils(
    grid_file="grid_file",
    coil_file="coil_file",
    current_file="current_file",
    field_file="field_file",
    field_periodicity=1,
):
    """Read the input data from grid_file, coil_file and current_file.
    Then calculate the magnetic field components for the discretized grid given from the input files
    and write the results to the output file field_file.

    Args:
        grid_file (str, optional): Input file containing the parameters for a discretized grid. Defaults to "grid_file".
        coil_file (str, optional): Input file containing the coil geometry. Defaults to "coil_file".
        current_file (str, optional): Input file containing the currents of each coil. Defaults to "current_file".
        field_file (str, optional): Output file into which the magnetic field components and calculation parameters are written to. Defaults to "field_file".
        field_periodicity (int, optional): Periodicity of the field in phi direction used for Tokamaks. Defaults to 1.
    """
    coils = read_coils(coil_file)

    currents = read_currents(current_file)

    grid = read_grid(grid_file, field_periodicity)

    BR, Bphi, BZ = get_field_on_grid(grid, coils, currents)

    write_field_to_file(field_file, grid, BR, Bphi, BZ, field_periodicity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid_file",
        type=str,
        default="grid_file",
        help="Input file containing the parameters for a discretized grid.",
    )
    parser.add_argument(
        "--coil_file",
        type=str,
        default="coil_file",
        help="Input file containing the coil geometry.",
    )
    parser.add_argument(
        "--current_file",
        type=str,
        default="current_file",
        help="Input file containing the currents of each coil.",
    )
    parser.add_argument(
        "--field_file",
        type=str,
        default="field_file",
        help="Output file into which the magnetic field components and calculation parameters are written to.",
    )
    parser.add_argument(
        "--field_periodicity",
        type=int,
        default=1,
        help="Periodicity of the field in phi direction used for Tokamaks.",
    )
    args = parser.parse_args()
    make_field_file_from_coils(
        args.grid_file,
        args.coil_file,
        args.current_file,
        args.field_file,
        args.field_periodicity,
    )
