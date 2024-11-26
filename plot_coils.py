# %%
from numpy import loadtxt
import matplotlib.pyplot as plt


def plot_coils():
    """Make a 3D plot of the coils. The geometry of the coils is read from co_asd.dd."""
    data = loadtxt("co_asd.dd", skiprows=1, usecols=(0, 1, 2, 4))
    ncoil = data[-1, 3].astype(int)  # number of coils
    nseg = (data.shape[0] / ncoil).astype(int)  # size of each coil segment

    # Split the data into X, Y, Z coordinates for each coil
    X = data[:, 0].reshape((ncoil, nseg))
    Y = data[:, 1].reshape((ncoil, nseg))
    Z = data[:, 2].reshape((ncoil, nseg))

    ax = plt.figure().add_subplot(projection="3d")
    for k in range(ncoil):
        # Plot line of the k-th coil
        ax.plot(X[k, :], Y[k, :], Z[k, :], "-k")
    plt.show()
