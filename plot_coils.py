import argparse
import matplotlib.pyplot as plt
from biotsavart import read_coils


def plot_coils(coil_file, figsize=(6, 5)):
    """Make a 3D plot of the coils. The geometry of the coils is read from coil_file.

    Args:
        coil_file (str): File name of the coil geometry
        figsize (tuple, optional): Size of the figure in inches. Defaults to (6, 5).
    """
    coil_parameters = read_coils(coil_file)
    n_coils = coil_parameters.coil_number[-1]  # total number of coils
    nseg = int(coil_parameters.n_nodes / n_coils)  # size of each coil segment

    # Split the data into X, Y, Z coordinates for each coil
    X = coil_parameters.X.reshape((n_coils, nseg))
    Y = coil_parameters.Y.reshape((n_coils, nseg))
    Z = coil_parameters.Z.reshape((n_coils, nseg))

    ax = plt.figure(figsize=figsize).add_subplot(projection="3d")
    for k in range(n_coils):
        # Plot line of the k-th coil
        ax.plot(X[k,], Y[k, :], Z[k, :], "-k")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coil_file",
        type=str,
        default="co_asd.dd",
        help="Input file containing the coil geometry.",
    )
    parser.add_argument(
        "--figsize", type=tuple, default=(6, 5), help="Size of the figure in inches"
    )
    args = parser.parse_args()
    plot_coils(args.coil_file, args.figsize)
