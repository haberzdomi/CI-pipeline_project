import argparse
from bdivfree import get_A_field_modes, calc_B_field_modes
from grid import grid
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from numpy import amin, amax, ceil, conj, empty, loadtxt, log10


def read_field(fname):
    """From fname get the discretized 3D-grid and the magnetic field components for each point on this grid.

    Args:
        fname (str): File name of the output of the biosavart_asdex calculation.

    Returns:
        grid (grid_parameters): Object containing the cylindrical 3D-grid and its parameters.
        BR (array[float], shape=(nR, nphi, nZ)): R-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
        Bphi (array[float], shape=(nR, nphi, nZ)): phi-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
        BZ (array[float], shape=(nR, nphi, nZ)): Z-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.


    """

    with open(fname, "r") as f:
        nR, nphi, nZ, _ = [int(data) for data in f.readline().split()]
        R_min, R_max = [float(data) for data in f.readline().split()]
        phi_min, phi_max = [float(data) for data in f.readline().split()]
        Z_min, Z_max = [float(data) for data in f.readline().split()]

    g = grid(nR, nphi, nZ, R_min, R_max, phi_min, phi_max, Z_min, Z_max)

    field_data = loadtxt(fname, skiprows=4)

    BR = field_data[:, 0].reshape(g.nR, g.nphi, g.nZ)
    Bphi = field_data[:, 1].reshape(g.nR, g.nphi, g.nZ)
    BZ = field_data[:, 2].reshape(g.nR, g.nphi, g.nZ)

    return g, BR, Bphi, BZ


def plot_modes(fname, n_modes, figsize=(8, 4)):
    """Get the grid and magnetic field components from fname and use fourier transformation
    to calculate the first 'n_modes' modes of R- and Z-component of the magnetic field.
    To ensure a divergence free field, the vector potential is determined and a spline
    interpolation is used to get points in between the grid for it. Its phi component is
    calculated such that the magnetic field gets divergency free in every point. The decadic
    logarithm of the square of the norm of the total magnetic field is caluculated for each
    of the modes on a grid which has double the resolution of the grid from fname. For each
    mode one subplot is created. The figure has a maximum of four subplots in a row, then a
    next row starts.

    Args:
        fname (str): File name of the output of the biotsavart_asdex calculation.
        n_modes (int): Number of modes which should be plotted.
        figsize (tuple, optional): Size of the figure in inches.
    """

    g, BR, Bphi, BZ = read_field(fname)

    # Evaluating the magnetic field modes on a grid with double the resolution of the B-field grid
    g_double = grid(
        2 * g.nR - 1,
        g.nphi,
        2 * g.nZ - 1,
        g.R_min,
        g.R_max,
        g.phi_min,
        g.phi_max,
        g.Z_min,
        g.Z_max,
    )

    A = get_A_field_modes(g, BR, Bphi, BZ)

    # Get logaritmic value of the squared norm of the magnetic field for each mode, i.e. for each subplot k
    log_Bn2 = empty((n_modes, g_double.nR, g_double.nZ))
    for k in range(n_modes):
        BnR, Bnphi, BnZ = calc_B_field_modes(g_double.R, g_double.Z, k + 1, A)
        # Add up the squared norm of the B-field components and take the logarithm of it
        log_Bn2[k, :, :] = log10(
            (BnR * conj(BnR) + Bnphi * conj(Bnphi) + BnZ * conj(BnZ)).real
        )

    color_norm = Normalize(vmin=amin(log_Bn2), vmax=amax(log_Bn2))

    # Create a figure with one subplot for each mode
    max_plots_per_row = 4
    n_cols = min(max_plots_per_row, n_modes)
    n_rows = ceil(n_modes / max_plots_per_row).astype(int)
    fig = plt.figure(layout="constrained", figsize=figsize)
    axs = fig.subplots(n_rows, n_cols).ravel()
    for k in range(n_modes):
        im = axs[k].imshow(
            log_Bn2[k, :, :].T,
            origin="lower",
            cmap="magma",
            extent=[g.R_min, g.R_max, g.Z_min, g.Z_max],
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
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fname",
        default="field_file",
        help="Output file containing the magnetic field components and calculation parameters.",
    )
    parser.add_argument(
        "--n_modes", default=8, help="Number of modes which should be plotted"
    )
    parser.add_argument(
        "--figsize", default=(8, 4), help="Size of the figure in inches"
    )
    args = parser.parse_args()

    plot_modes(args.fname, args.n_modes, args.figsize)
