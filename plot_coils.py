import argparse
import matplotlib.pyplot as plt
from numpy import loadtxt


def plot_coils(fname, figsize=(6, 5)):
    """Make a 3D plot of the coils. The geometry of the coils is read from fname.

    Args:
        fname (str): File name of the coil geometry
        figsize (tuple, optional): Size of the figure in inches
    """
    data = loadtxt(fname, skiprows=1, usecols=(0, 1, 2, 4))
    ncoil = data[-1, 3].astype(int)  # number of coils
    nseg = (data.shape[0] / ncoil).astype(int)  # size of each coil segment

    # Split the data into X, Y, Z coordinates for each coil
    X = data[:, 0].reshape((ncoil, nseg))
    Y = data[:, 1].reshape((ncoil, nseg))
    Z = data[:, 2].reshape((ncoil, nseg))

    ax = plt.figure(figsize=figsize).add_subplot(projection="3d")
    for k in range(ncoil):
        # Plot line of the k-th coil
        ax.plot(X[k, :], Y[k, :], Z[k, :], "-k")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fname", default="coil_file", help="Input file containing the coil geometry."
    )
    parser.add_argument(
        "--figsize", default=(6, 5), help="Size of the figure in inches"
    )
    args = parser.parse_args()
    plot_coils(args.fname, args.figsize)
