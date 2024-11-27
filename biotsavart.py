# Dominik Haberz 11916636
# Clara Rinner 01137166
# %%
# program biotsavart
import numpy as np
from grid import grid
import argparse


class coils:
    def __init__(self, X, Y, Z, has_current, coil_number, n_nodes):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.has_current = has_current
        self.coil_number = coil_number
        self.n_nodes = n_nodes


def calc_biotsavart(grid_coordinates, coils, currents):
    """
    Calculate the magnetic field components by evaluating the Biot-Savart integral.

    Args:
        grid_coordinates (array[float], shape=(3)): cylindrical coordinates of the grid point
        coil_data: coordinates of the coil points

    Returns:
        BRI (float): radial component fo the magnetic field
        BfI (float): toroidal component of the magnetic field
        BZI (float): axial component of the magnetic field
    """

    RI = grid_coordinates[0]
    fI = grid_coordinates[1]
    ZI = grid_coordinates[2]
    cosf = np.cos(fI)
    sinf = np.sin(fI)
    Y_grid = RI * sinf
    X_grid = RI * cosf
    Z_grid = ZI
    grid_point = [X_grid, Y_grid, Z_grid]

    B = [0, 0, 0]

    coil_point = [coils.X[0], coils.Y[0], coils.Z[0]]
    # Difference of grid point and first coil point | r-r'[0]
    R1_vector = np.subtract(grid_point, coil_point)
    # Distance between grid point and first coil point | |r-r'[0]|
    R1 = np.linalg.norm(R1_vector)

    for K in range(1, coils.n_nodes):  # loops through the remaining coil points

        coil_point_previous = coil_point
        coil_point = [coils.X[K], coils.Y[K], coils.Z[K]]

        # Difference between the current coil point and the previous one | l
        L = np.subtract(coil_point, coil_point_previous)
        # Difference between grid point and current coil point | r-r'[k]
        R2_vector = np.subtract(grid_point, coil_point)

        scalar_product = np.dot(L, R2_vector)  # Scalar product of l and r-r'[k]
        # Distance between grid point and current coil point | |r-r'[k]|
        R2 = np.linalg.norm(R2_vector)

        if coils.has_current[K - 1] != 0.0:  # If not first point of a coil

            OBCP = 1.0 / (R2 * (R1 + R2) + scalar_product)
            FAZRDA = -(R1 + R2) * OBCP / R1 / R2 * currents[coils.coil_number[K] - 1]
            B_B1 = np.cross(R2_vector, L)

            B = np.add(np.dot(B_B1, FAZRDA), B)

        R1 = R2

    B_R = B[0] * cosf + B[1] * sinf
    B_phi = B[1] * cosf - B[0] * sinf
    B_Z = B[2]

    return B_R, B_phi, B_Z


def read_coils(coil_file):
    """
    Return the data in coil_file as a coils object.

    Args:
        None
    Returns:
        coils object with the following attributes:
            X, Y, Z - coodinate lists of coil points
            has_current - 0 if last point in a coil, otherwise 1
            coil_number - coil number which the point belongs to
            n_nodes - total number of coil points
    """

    file = open(coil_file, "r")
    n_nodes = int(file.readline())

    data = np.loadtxt(file)
    X = data[:, 0]
    Y = data[:, 1]
    Z = data[:, 2]
    has_current = data[:, 3]
    coil_number = [int(d) for d in data[:, 4]]

    # Raise error if the current factor for the last node of the last coil is not 0.
    if has_current[n_nodes - 1] != 0.0:
        raise Exception(str(has_current[n_nodes - 1]) + "=cbc~=0, stop")

    file.close()
    return coils(X, Y, Z, has_current, coil_number, n_nodes)


def read_currents(current_file):
    file = open(current_file, "r")
    currents = [float(data) for data in file.read().split()]
    file.close()
    return currents


def read_grid(grid_file, field_periodicity):
    f1 = open(grid_file, "r")
    nR, nphi, nZ = [int(data) for data in f1.readline().split()]
    R_min, R_max = [float(data) for data in f1.readline().split()]
    phi_min = 0
    phi_max = 2 * np.pi / field_periodicity
    Z_min, Z_max = [float(data) for data in f1.readline().split()]
    f1.close()

    return grid(nR, nphi, nZ, R_min, R_max, phi_min, phi_max, Z_min, Z_max)


def write_field_to_file(field_file, grid, B, field_periodicity):
    file = open(field_file, "w")
    # Write the input parameters for the magnetic field calculation to the output file.
    file.write(f"{grid.nR} {grid.nphi} {grid.nZ} {field_periodicity}\n")
    file.write(f"{grid.R_min} {grid.R_max}\n")
    file.write(f"{grid.phi_min} {grid.phi_max}\n")
    file.write(f"{grid.Z_min} {grid.Z_max}\n")
    # Loop over the grid points and calculate the magnetic field components.
    for b in B:
        file.write(f"{b[0]} {b[1]} {b[2]}\n")
    file.close()


def get_field_on_grid(grid, coils, currents):
    B = []
    for r in grid.R:
        for p in grid.phi:
            for z in grid.Z:
                x = [r, p, z]  # coordinates for current grid point
                B_R, B_phi, B_Z = calc_biotsavart(x, coils, currents)
                B.append([B_R, B_phi, B_Z])
    return B


def make_field_file_from_coils(
    grid_file="grid_file",
    coil_file="coil_file",
    current_file="current_file",
    field_file="field_file",
    field_periodicity=1,
):
    coils = read_coils(coil_file)
    currents = read_currents(current_file)
    #
    grid = read_grid(grid_file, field_periodicity)
    #
    B = get_field_on_grid(grid, coils, currents)
    #
    write_field_to_file(field_file, grid, B, field_periodicity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid_file",
        type=str,
        default="grid_file",
        help="Input file containing the grid information.",
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
        help="Output file containing the magnetic field components and calculation parameters.",
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
