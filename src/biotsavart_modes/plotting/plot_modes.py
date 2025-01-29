import argparse
from biotsavart_modes.helpers.bdivfree import get_A_field_modes, calc_B_field_modes
from biotsavart_modes.helpers.grid import GRID
import h5py
from importlib.resources import files
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
from pathlib import WindowsPath
from os.path import exists


def read_field(field_file):
    """From field_file get the discretized 3D-grid and the magnetic field components for each point on this grid.

    Args:
        field_file (WindowsPath): File of the magnetic field calculation output. If the file does not exist, the file "field_original.dat" is taken.

    Returns:
        grid (GRID object): Object containing the cylindrical 3D-grid and its parameters.
        BR (array[float], shape=(nR, nphi, nZ)): R-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
        Bphi (array[float], shape=(nR, nphi, nZ)): phi-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
        BZ (array[float], shape=(nR, nphi, nZ)): Z-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
    """
    if not exists(field_file):
        field_file = "tests/golden_record/field.dat"

    with open(field_file, "r") as f:
        nR, nphi, nZ, _ = [int(data) for data in f.readline().split()]
        R_min, R_max = [float(data) for data in f.readline().split()]
        phi_min, phi_max = [float(data) for data in f.readline().split()]
        Z_min, Z_max = [float(data) for data in f.readline().split()]

    grid = GRID(nR, nphi, nZ, R_min, R_max, phi_min, phi_max, Z_min, Z_max)

    field_data = np.loadtxt(field_file, skiprows=4)

    BR = field_data[:, 0].reshape(grid.nR, grid.nphi, grid.nZ)
    Bphi = field_data[:, 1].reshape(grid.nR, grid.nphi, grid.nZ)
    BZ = field_data[:, 2].reshape(grid.nR, grid.nphi, grid.nZ)

    return grid, BR, Bphi, BZ


def read_field_hdf5(field_file):
    """From the HDF5-file field_file get the discretized 3D-grid and the magnetic field components for each point on this grid.

    Args:
        field_file (WindowsPath): File of the magnetic field calculation output.
                                  If the file does not exist, the file "field_original.dat" is read instead by calling read_field.
                                  Must have a HDF5 file extension.

    Returns:
        grid (GRID object): Object containing the cylindrical 3D-grid and its parameters.
        BR (array[float], shape=(nR, nphi, nZ)): R-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
        Bphi (array[float], shape=(nR, nphi, nZ)): phi-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
        BZ (array[float], shape=(nR, nphi, nZ)): Z-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
    """
    if not exists(field_file):
        field_file = "tests/golden_record/field.dat"
        return read_field(field_file)
    field_fname = field_file
    if not (field_fname.endswith(".h5") or field_fname.endswith(".hdf5")):
        raise ValueError(
            "Output file must be a HDF5 file. Instead call read_field_netcdf for netCDF4 files or read_field for .dat files."
        )

    f = h5py.File(field_file, "r")
    nR, nphi, nZ, R_min, R_max, Z_min, Z_max = f["grid/input_parameters"][()]

    grid = GRID(int(nR), int(nphi), int(nZ), R_min, R_max, 0, 2 * np.pi, Z_min, Z_max)

    BR, Bphi, BZ = f["magnetic_field/data"][()]
    BR = BR.reshape(grid.nR, grid.nphi, grid.nZ)
    Bphi = Bphi.reshape(grid.nR, grid.nphi, grid.nZ)
    BZ = BZ.reshape(grid.nR, grid.nphi, grid.nZ)

    f.close()
    return grid, BR, Bphi, BZ


def read_field_netcdf(field_file):
    """From the netCDF4-file field_file get the discretized 3D-grid and the magnetic field components for each point on this grid.

    Args:
        field_file (WindowsPath): File of the magnetic field calculation output.
                                  If the file does not exist, the file "field_original.dat" is read instead by calling read_field.
                                  Must have a netCDF4 file extension.

    Returns:
        grid (GRID object): Object containing the cylindrical 3D-grid and its parameters.
        BR (array[float], shape=(nR, nphi, nZ)): R-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
        Bphi (array[float], shape=(nR, nphi, nZ)): phi-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
        BZ (array[float], shape=(nR, nphi, nZ)): Z-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
    """
    if not exists(field_file):
        field_file = "tests/golden_record/field.dat"
        return read_field(field_file)
    field_fname = field_file
    if not (field_fname.endswith(".nc") or field_fname.endswith(".cdf")):
        raise ValueError(
            "Output file must be a netCDF4 file. Instead call read_field_hdf5 for HDF5 files or read_field for .dat files."
        )

    root_grp = Dataset(field_file, "r", format="NETCDF4")
    grid_grp = root_grp["grid"]
    grid_grp
    nR = len(grid_grp["R"])
    nphi = len(grid_grp["phi"])
    nZ = len(grid_grp["Z"])
    R_min = float(grid_grp["R"][0])
    R_max = float(grid_grp["R"][-1])
    Z_min = float(grid_grp["Z"][0])
    Z_max = float(grid_grp["Z"][-1])

    grid = GRID(nR, nphi, nZ, R_min, R_max, 0, 2 * np.pi, Z_min, Z_max)

    field_grp = root_grp["magnetic_field"]
    BR = field_grp["BR"][:].reshape(grid.nR, grid.nphi, grid.nZ)
    Bphi = field_grp["Bphi"][:].reshape(grid.nR, grid.nphi, grid.nZ)
    BZ = field_grp["BZ"][:].reshape(grid.nR, grid.nphi, grid.nZ)

    root_grp.close()

    # filled(np.nan) is used to convert masked arrays to normal arrays
    return grid, BR.filled(np.nan), Bphi.filled(np.nan), BZ.filled(np.nan)


def plot_modes(field_file, n_modes, figsize=(8, 4)):
    """Get the grid and magnetic field components from field_file and use fourier transformation
    to calculate the first 'n_modes' modes of R- and Z-component of the magnetic field.
    To ensure a divergence free field, the vector potential is determined and a spline
    interpolation is used to get points in between the grid for it. Its phi component is
    calculated such that the magnetic field gets divergency free in every point. The decadic
    logarithm of the square of the norm of the total magnetic field is caluculated for each
    of the modes on a grid which has double the resolution of the grid from field_file. For each
    mode one subplot is created. The figure has a maximum of four subplots in a row, then a
    next row starts.

    Args:
        field_file (WindowsPath): File of the magnetic field calculation output.
        n_modes (int): Number of modes which should be plotted.
        figsize (tuple, optional): Size of the figure in inches. Defaults to (8, 4).
    """
    field_fname = field_file
    if field_fname.endswith(".h5") or field_fname.endswith(".hdf5"):
        grid, BR, Bphi, BZ = read_field_hdf5(field_file)
    elif field_fname.endswith(".nc") or field_fname.endswith(".cdf"):
        grid, BR, Bphi, BZ = read_field_netcdf(field_file)
    else:
        grid, BR, Bphi, BZ = read_field(field_file)

    # Evaluating the magnetic field modes on a grid with double the resolution of the B-field grid
    grid_double = GRID(
        2 * grid.nR - 1,
        grid.nphi,
        2 * grid.nZ - 1,
        grid.R_min,
        grid.R_max,
        grid.phi_min,
        grid.phi_max,
        grid.Z_min,
        grid.Z_max,
    )

    if grid_double.nphi - 1 < 2 * n_modes:
        raise ValueError("n_modes is too high. Condition: 2 * n_max <= nphi - 1")

    A = get_A_field_modes(grid, BR, Bphi, BZ)

    # Get logaritmic value of the squared norm of the magnetic field for each mode, i.e. for each subplot k
    log_Bn2 = np.empty((n_modes, grid_double.nR, grid_double.nZ))
    for k in range(n_modes):
        BnR, Bnphi, BnZ = calc_B_field_modes(grid_double.R, grid_double.Z, k + 1, A)
        # Add up the squared norm of the B-field components and take the logarithm of it
        log_Bn2[k, :, :] = np.log10(
            (BnR * np.conj(BnR) + Bnphi * np.conj(Bnphi) + BnZ * np.conj(BnZ)).real
        )

    color_norm = Normalize(vmin=np.amin(log_Bn2), vmax=np.amax(log_Bn2))

    # Create a figure with one subplot for each mode
    max_plots_per_row = 4
    n_cols = min(max_plots_per_row, n_modes)
    n_rows = np.ceil(n_modes / max_plots_per_row).astype(int)
    fig = plt.figure(layout="constrained", figsize=figsize)
    axs = fig.subplots(n_rows, n_cols).ravel()
    for k in range(n_modes):
        im = axs[k].imshow(
            log_Bn2[k, :, :].T,
            origin="lower",
            cmap="magma",
            extent=[grid.R_min, grid.R_max, grid.Z_min, grid.Z_max],
        )
        im.set_norm(color_norm)
        axs[k].set_title(f"$n = {k + 1}$")

    # Remove remaining empty subplots
    k = n_modes
    while k < n_rows * n_cols:
        axs[k].axis("off")
        k += 1

    # Add colorbar and labels
    cbar = fig.colorbar(im, ax=axs, location="right")
    cbar.set_label(r"$\log_{10} |\vec{B}_{n}|^{2}$")
    # Save fig at the same location where the field file is located
    plt.savefig("field_modes.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--field_file",
        type=WindowsPath,
        default="src/biotsavart_modes/output/field.h5",
        help="File of the magnetic field calculation output containing the magnetic field components and calculation parameters.",
    )
    parser.add_argument(
        "--n_modes", type=int, default=8, help="Number of modes which should be plotted"
    )
    parser.add_argument(
        "--figsize", type=tuple, default=(8, 4), help="Size of the figure in inches"
    )
    args = parser.parse_args()

    plot_modes(args.field_file, args.n_modes, args.figsize)
