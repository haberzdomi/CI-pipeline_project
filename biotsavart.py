# Dominik Haberz 11916636
# Clara Rinner 01137166
#
import argparse
from grid import GRID
import h5py
from netCDF4 import Dataset
import numba as nb
import numpy as np
from timeit import default_timer


##
@nb.experimental.jitclass(
    [
        ("X", nb.float64[:]),
        ("Y", nb.float64[:]),
        ("Z", nb.float64[:]),
        ("has_current", nb.float64[:]),
        ("coil_number", nb.int32[:]),
        ("n_nodes", nb.int32),
    ]
)
class COILS:
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    has_current: np.ndarray
    coil_number: np.ndarray
    n_nodes: int

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


@nb.njit
def calc_biotsavart(grid_coordinates, coils, currents):
    """
    Calculate the magnetic field components by evaluating the Biot-Savart integral for the
    given coil geometry and currents. This function loops over all coil points.

    Args:
        grid_coordinates (array[float], shape=(3, )): cylindrical coordinates of the grid point
        coils (coils object): coil data
        currents (array[float],shape=(n_coils, )): Currents of each coil. n_coils is the total number of coils
    Returns:
        BR (float): Radial component fo the magnetic field
        Bphi (float): Toroidal component of the magnetic field
        BZ (float): Axial component of the magnetic field
    """

    R_grid = grid_coordinates[0]
    phi_grid = grid_coordinates[1]

    cos_phi = np.cos(phi_grid)
    sin_phi = np.sin(phi_grid)
    Y_grid = R_grid * sin_phi
    X_grid = R_grid * cos_phi
    Z_grid = grid_coordinates[2]
    grid_point = np.array([X_grid, Y_grid, Z_grid])

    B = np.array([0.0, 0.0, 0.0])

    coil_point = np.array([coils.X[0], coils.Y[0], coils.Z[0]])

    # Distance between grid point and first coil point: |r-r'[0]|
    R1_vector = np.subtract(grid_point, coil_point)
    R1 = np.linalg.norm(R1_vector)

    for K in range(1, coils.n_nodes):  # loops through the remaining coil points
        coil_point_previous = coil_point
        coil_point = np.array([coils.X[K], coils.Y[K], coils.Z[K]])

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

            B = cross_product * factor2 + B

        R1 = R2

    BR = B[0] * cos_phi + B[1] * sin_phi
    Bphi = B[1] * cos_phi - B[0] * sin_phi
    BZ = B[2]

    return BR, Bphi, BZ


@nb.njit
def calc_biotsavart_vectorized(grid_coordinates, coils, currents):
    """
    Calculate the magnetic field components by evaluating the Biot-Savart integral for the given
    coil geometry and currents. This function uses vectorized operations to deal with the coil points.

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
    cos_phi = np.cos(fI)
    sin_phi = np.sin(fI)
    Y_grid = RI * sin_phi
    X_grid = RI * cos_phi
    Z_grid = ZI
    grid_point = np.array([X_grid, Y_grid, Z_grid])

    coil_points = np.zeros((coils.n_nodes, 3))
    coil_points[:, 0] = coils.X
    coil_points[:, 1] = coils.Y
    coil_points[:, 2] = coils.Z
    R_vectors = np.subtract(grid_point, coil_points)

    R = np.sqrt(R_vectors[:, 0] ** 2 + R_vectors[:, 1] ** 2 + R_vectors[:, 2] ** 2)

    L_vectors = np.subtract(coil_points[1:], coil_points[:-1])

    scalar_products = (
        L_vectors[:, 0] * R_vectors[1:][:, 0]
        + L_vectors[:, 1] * R_vectors[1:][:, 1]
        + L_vectors[:, 2] * R_vectors[1:][:, 2]
    )

    current_array = np.array([currents[i - 1] for i in coils.coil_number[1:]])

    factor1_array = 1 / (R[1:] * (R[:-1] + R[1:]) + scalar_products)
    factor2_array = (
        -(R[:-1] + R[1:])
        * factor1_array
        / R[:-1]
        / R[1:]
        * coils.has_current[:-1]
        * current_array
    )
    cross_products = np.cross(R_vectors[1:], L_vectors)

    B = np.sum(cross_products * factor2_array[:, np.newaxis], axis=0)

    B_R = B[0] * cos_phi + B[1] * sin_phi
    B_phi = B[1] * cos_phi - B[0] * sin_phi
    B_Z = B[2]

    return B_R, B_phi, B_Z


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
    coil_number = data[:, 4].astype(int)

    last_nodes_idxs = np.where(np.diff(coil_number) != 0)[0]

    # Raise error if the current factor for the last node of each coil is not 0.
    if np.any(has_current[last_nodes_idxs] != 0.0):
        raise Exception(str(has_current[n_nodes - 1]) + "=cbc~=0, stop")

    return COILS(X, Y, Z, has_current, coil_number, n_nodes)


def read_currents(current_file):
    """Return the input data stored in current_file.

    Args:
        current_file (str): Input file containing the currents of each coil.
    Returns:
        currents (array[float], shape=(n_coils, )): Currents of each coil. n_coils is the total number of coils.

    """
    with open(current_file, "r") as f:
        currents = np.array([float(data) for data in f.readline().split()])
    return currents


def read_grid(grid_file, field_periodicity=1):
    """Read the input data stored in grid_file and return a GRID object.

    Args:
        grid_file (str): Input file containing the parameters for a discretized grid.
        field_periodicity (int): Periodicity of the field in phi direction used for Tokamaks. Defaults to 1.

    Returns:
        GRID object with the parameters read from the grid_file.
    """
    with open(grid_file, "r") as f:
        nR, nphi, nZ = [int(data) for data in f.readline().split()]
        R_min, R_max = [float(data) for data in f.readline().split()]
        Z_min, Z_max = [float(data) for data in f.readline().split()]
    phi_min = 0
    phi_max = 2 * np.pi / field_periodicity

    return GRID(nR, nphi, nZ, R_min, R_max, phi_min, phi_max, Z_min, Z_max)


def get_field_on_grid(grid, coils, currents, integrator):
    """Loop over the discretized grid points and calculate the magnetic field components.

    Args:
        grid (GRID object): Object containing the cylindrical 3D-grid and its parameters.
        coils (coils object): coil data
        currents (array[float], shape=(n_coils, )): Currents of each coil. n_coils is the total number of coils.
        integrator (function): Function to evaluate the Biot-Savart integral and calculate the magnetic field components.
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
                x = np.array([r, p, z])
                BR[i], Bphi[i], BZ[i] = integrator(x, coils, currents)
                i += 1
    return BR, Bphi, BZ


@nb.njit(parallel=True)
def get_field_on_grid_numba_parallel(grid, coils, currents, integrator):
    """Loop parallelized over the discretized grid points and calculate the magnetic field components.

    Args:
        grid (GRID object): Object containing the cylindrical 3D-grid and its parameters.
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

    for j in nb.prange(grid.nR):
        for k in nb.prange(grid.nphi):
            for l in nb.prange(grid.nZ):
                x = np.array([grid.R[j], grid.phi[k], grid.Z[l]])
                i = j * grid.nphi * grid.nZ + k * grid.nZ + l
                BR[i], Bphi[i], BZ[i] = integrator(x, coils, currents)

    return BR, Bphi, BZ


def write_field_to_file(field_file, grid, BR, Bphi, BZ, field_periodicity):
    """Write the calculation parameters (grid, field_periodicity) and the magnetic field components to the output file.

    Args:
        field_file (str): Output file into which the magnetic field components and calculation parameters are written to.
        grid (GRID object): Object containing the cylindrical 3D-grid and its parameters.
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


def write_field_hdf5(field_file, grid, BR, Bphi, BZ, field_periodicity):
    """Write the calculation parameters (grid, field_periodicity) and the magnetic field components to an output file of the format "HDF5".

    Args:
        field_file (str): Output file into which the magnetic field components and calculation parameters are written to. Must have a HDF5 file extension.
        grid (GRID object): Object containing the cylindrical 3D-grid and its parameters.
        BR (array[float], shape=(n_points, )): Radial component fo the magnetic field. n_points is the total number of grid points.
        Bphi (array[float], shape=(n_points, )): Toroidal component of the magnetic field. n_points is the total number of grid points.
        BZ (array[float], shape=(n_points, )): Axial component of the magnetic field. n_points is the total number of grid points.
        field_periodicity (int): Periodicity of the field in phi direction used for Tokamaks.
    """
    if not (field_file.endswith(".h5") or field_file.endswith(".hdf5")):
        raise ValueError(
            "Output file must be a HDF5 file when selecting the write_field_hdf5 writer."
        )
    f = h5py.File(field_file, "w")

    grid_grp = f.create_group("grid")
    grid_data = np.array(
        [grid.nR, grid.nphi, grid.nZ, grid.R_min, grid.R_max, grid.Z_min, grid.Z_max]
    )
    grid_grp.create_dataset("input_parameters", (7,), data=grid_data)
    grid_grp["input_parameters"].attrs[
        "description"
    ] = "nR, nphi, nZ, R_min, R_max, Z_min, Z_max"
    grid_grp["input_parameters"].attrs["units"] = " - , - , - , cm, cm, cm, cm"
    grid_grp.create_dataset("periodicity", (1,), data=field_periodicity)
    grid_grp["periodicity"].attrs[
        "description"
    ] = "Periodicity of the field in phi direction used for Tokamaks."
    grid_grp["periodicity"].attrs["units"] = " "

    field_grp = f.create_group("magnetic_field")
    dimension = (3, grid.nR * grid.nphi * grid.nZ)
    field_grp.create_dataset("data", dimension, data=[BR, Bphi, BZ])
    field_grp["data"].attrs["description"] = "BR, Bphi, BZ"
    field_grp["data"].attrs["units"] = "G, G, G"

    f.close()


def write_field_netcdf(field_file, grid, BR, Bphi, BZ, field_periodicity):
    """Write the calculation parameters (grid, field_periodicity) and the magnetic field components to an output file of the format "netCDF4".

    Args:
        field_file (str): Output file into which the magnetic field components and calculation parameters are written to. Must have a netCDF4 file extension.
        grid (GRID object): Object containing the cylindrical 3D-grid and its parameters.
        BR (array[float], shape=(n_points, )): Radial component fo the magnetic field. n_points is the total number of grid points.
        Bphi (array[float], shape=(n_points, )): Toroidal component of the magnetic field. n_points is the total number of grid points.
        BZ (array[float], shape=(n_points, )): Axial component of the magnetic field. n_points is the total number of grid points.
        field_periodicity (int): Periodicity of the field in phi direction used for Tokamaks.
    """
    if not (field_file.endswith(".nc") or field_file.endswith(".cdf")):
        raise ValueError(
            "Output file must be a netCDF4 file when selecting the write_field_netcdf writer."
        )

    root_grp = Dataset("field.nc", "w", format="NETCDF4")

    root_grp.createDimension("R", grid.nR)
    root_grp.createDimension("phi", grid.nphi)
    root_grp.createDimension("Z", grid.nZ)
    root_grp.createDimension("B", grid.nR * grid.nphi * grid.nZ)

    grid_grp = root_grp.createGroup("grid")
    grid_grp.createVariable("R", "f8", ("R",))
    grid_grp["R"].unit = "cm"
    grid_grp["R"][:] = grid.R
    grid_grp.createVariable("phi", "f8", ("phi",))
    grid_grp["phi"].unit = "rad"
    grid_grp["phi"][:] = grid.phi
    grid_grp.createVariable("Z", "f8", ("Z",))
    grid_grp["Z"].unit = "cm"
    grid_grp["Z"][:] = grid.Z
    grid_grp.createVariable("periodicity", "f8")
    grid_grp["periodicity"].description = (
        "Periodicity of the field in phi direction used for Tokamaks."
    )
    grid_grp["periodicity"].units = " "
    grid_grp["periodicity"][:] = field_periodicity

    field_grp = root_grp.createGroup("magnetic_field")
    field_grp.createVariable("BR", "f8", ("B",))
    field_grp["BR"].description = "Radial component of the magnetic field"
    field_grp["BR"].unit = "G"
    field_grp["BR"][:] = BR
    field_grp.createVariable("Bphi", "f8", ("B",))
    field_grp["Bphi"].description = "Toroidal component of the magnetic field"
    field_grp["Bphi"].unit = "G"
    field_grp["Bphi"][:] = Bphi
    field_grp.createVariable("BZ", "f8", ("B",))
    field_grp["BZ"].description = "Axial component of the magnetic field"
    field_grp["BZ"].unit = "G"
    field_grp["BZ"][:] = BZ

    root_grp.close()


def make_field_file_from_coils(
    grid_file="biotsavart.inp",
    coil_file="co_asd.dd",
    current_file="cur_asd.dd",
    field_file="field.h5",
    integrator=calc_biotsavart,
    grid_iterator=get_field_on_grid_numba_parallel,
    field_periodicity=1,
):
    """Read the input data from grid_file, coil_file and current_file. Then calculate the
    magnetic field components for the discretized grid given from the input files and write
    the results to the output file field_file. Print the time it took to calculate the field.

    Args:
        grid_file (str, optional): Input file containing the parameters for a discretized grid. Defaults to "biotsavart.inp".
        coil_file (str, optional): Input file containing the coil geometry. Defaults to "co_asd.dd".
        current_file (str, optional): Input file containing the currents of each coil. Defaults to "cur_asd.dd".
        field_file (str, optional): Output file into which the magnetic field components and calculation parameters are written to. Defaults to "field.h5".
        integrator (function, optional): Function to evaluate the Biot-Savart integral and calculate the magnetic field components. Defaults to calc_biotsavart.
        grid_iterator (function, optional): Function which iterates over the grid points onto which the magnetic field is calculated. Defaults to get_field_on_grid_numba_parallel.
        field_periodicity (int, optional): Periodicity of the field in phi direction used for Tokamaks. Defaults to 1.
    """

    try:
        open(coil_file).close
    except:
        print(f"{coil_file} not found")
    try:
        open(current_file).close
    except:
        print(f"{current_file} not found")
    try:
        open(grid_file).close
    except:
        print(f"{grid_file} not found")

    coils = read_coils(coil_file)

    currents = read_currents(current_file)

    if coils.coil_number[-1] != len(currents):
        raise StopIteration(
            "Number of coils needs to be the same as the number of currents"
        )

    grid = read_grid(grid_file, field_periodicity)
    if grid.R_min < 0 or grid.R_max < 0:
        raise ValueError("Radius has to be positive")
    if grid.R_min > grid.R_max:
        raise ValueError("R_min must be lower than R_max")
    if grid.Z_min > grid.Z_max:
        raise ValueError("Z_min must be lower than Z_max")

    start = default_timer()
    BR, Bphi, BZ = grid_iterator(grid, coils, currents, integrator)
    print(f"Field calculation took: {default_timer() - start} s")

    if field_file.endswith(".h5") or field_file.endswith(".hdf5"):
        write_field_hdf5(field_file, grid, BR, Bphi, BZ, field_periodicity)
    elif field_file.endswith(".nc") or field_file.endswith(".cdf"):
        write_field_netcdf(field_file, grid, BR, Bphi, BZ, field_periodicity)
    else:
        write_field_to_file(field_file, grid, BR, Bphi, BZ, field_periodicity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid_file",
        type=str,
        default="biotsavart.inp",
        help="Input file containing the parameters for a discretized grid.",
    )
    parser.add_argument(
        "--coil_file",
        type=str,
        default="co_asd.dd",
        help="Input file containing the coil geometry.",
    )
    parser.add_argument(
        "--current_file",
        type=str,
        default="cur_asd.dd",
        help="Input file containing the currents of each coil.",
    )
    parser.add_argument(
        "--field_file",
        type=str,
        default="field.h5",
        help="Output file into which the magnetic field components and calculation parameters are written to.",
    )
    parser.add_argument(
        "--integrator",
        type=str,
        default="calc_biotsavart_vectorized",
        choices=["calc_biotsavart", "calc_biotsavart_vectorized"],
        help="Name of the function to evaluate the Biot-Savart integral and calculate the magnetic field components",
    )
    parser.add_argument(
        "--grid_iterator",
        type=str,
        default="get_field_on_grid_numba_parallel",
        choices=["get_field_on_grid", "get_field_on_grid_numba_parallel"],
        help="Name of the function which iterates over the grid points onto which the magnetic field is calculated.",
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
        eval(args.integrator),
        eval(args.grid_iterator),
        args.field_periodicity,
    )
