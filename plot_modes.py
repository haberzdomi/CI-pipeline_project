#%%
from numpy import amin, amax, ceil, conj, empty, linspace, log10
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from bdivfree import field_c

def plot_modes(n_modes=8, figsize=(8, 4)):
    """
    Read the output of the biotsavart_asdex calculation ('field.dat') and its input 
    parameters ('biotsavart.inp'). Fourier transformation is used to calculate the first 
    'n_modes' modes of R- and Z-component of the magnetic field. A spline interpolation 
    is used to get points in between the grid used in field.dat. The phi component is
    calculated such that the magnetic field gets divergency free in every point. Precisely 
    the decadic logarithm of the square of the norm of the total magnetic field is 
    caluculated for each of the modes on a grid which has double the resolution of the 
    3D-grid defined for biotsavart_asdex.py. For each mode one subplot is created. The 
    figure has a maximum of four subplots in a row, then a next row starts. 
    """
    
    # TODO: Load grid via read_grid() function
    # Read biotsavart input parameters
    with open('biotsavart.inp', 'r') as f:
        # number of grid points for each dimension for discretization
        nR, nphi, nZ = [int(data) for data in f.readline().split()]
        # Boundaries of R- and Z-dimension
        R_min, R_max = [float(data) for data in f.readline().split()]
        Z_min, Z_max = [float(data) for data in f.readline().split()]

    # Double the resolution
    R = linspace(R_min, R_max, 2 * nR - 1)
    Z = linspace(Z_min, Z_max, 2 * nZ - 1)

    # Get logaritmic value of the squared norm of the magnetic field for each mode, i.e. for each subplot k
    log_Bn2 = empty((n_modes, 2 * nR - 1, 2 * nZ - 1))
    for k in range(n_modes):
        BnR, Bnphi, BnZ = field_c(R, Z, k + 1) # Get the magnetic field components
        # Add up the squared norm of the B-field components and take the logarithm of it
        log_Bn2[k, :, :] = log10((BnR * conj(BnR) +
                                Bnphi * conj(Bnphi) +
                                BnZ * conj(BnZ)).real)
    # color plot normalization 
    norm = Normalize(vmin=amin(log_Bn2), vmax=amax(log_Bn2))

    # Create a figure with one subplot for each mode
    max_plots_per_row = 4
    n_cols = min(max_plots_per_row, n_modes)
    n_rows = ceil(n_modes / max_plots_per_row).astype(int)
    fig = plt.figure(layout='constrained', figsize=figsize)
    axs = fig.subplots(n_rows, n_cols).ravel()
    for k in range(n_modes):
        im = axs[k].imshow(log_Bn2[k, :, :].T, origin='lower', cmap='magma',
                        extent=[R_min, R_max, Z_min, Z_max])
        im.set_norm(norm) # normalization of the colors of the image
        axs[k].set_title(f"$n = {k + 1}$")

    # Remove remaining empty subplots
    k=n_modes
    while k < n_rows*n_cols:
        axs[k].axis('off')
        k+=1

    # Add colorbar and labels
    cbar = fig.colorbar(im, ax=axs, location='right')
    cbar.set_label(r'$\log_{10} |\vec{B}_{n}|^{2}$')
    plt.show()