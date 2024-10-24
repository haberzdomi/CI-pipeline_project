#%%
from numpy import amin, amax, conj, empty, linspace, log10
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from bdivfree import field_c

"""
Read the output of the biotsavart_asdex calculation ('field,dat') and its input 
parameters ('biotsavart.inp'). Fourier transformation is used to calculate the 
first n modes of R- and Z-component of the magnetic field. The phi component is 
calculated such that the magnetic field gets divergency free. Precisely the decadic 
logarithm of the square of the norm of the total magnetic field is caluculated for 
each of the modes on a grid which has double the resolution of the 3D-grid defined 
for biotsavart_asdex.py. For each modem one colorplot is created.
"""

#%%

# Read biotsavart input parameters
with open('biotsavart.inp', 'r') as f:
    # number of grid points for each dimension for discretization
    nR, nphi, nZ = [int(data) for data in f.readline().split()]
    # Boundaries of R- and Z-dimension
    R_min, R_max = [float(data) for data in f.readline().split()]
    Z_min, Z_max = [float(data) for data in f.readline().split()]

# double the resolution
R = linspace(R_min, R_max, 2 * nR - 1)
Z = linspace(Z_min, Z_max, 2 * nZ - 1)

# vert*horz = number of modes plotted
vert = 2 # number of rows for subplots
horz = 4 # number of columns for subplots
log_Bn2 = empty((vert * horz, 2 * nR - 1, 2 * nZ - 1)) 

# Get logaritmic value of the squared norm of the magnetic field for each mode, i.e. for each subplot k
for k in range(vert * horz):
    BnR, Bnphi, BnZ = field_c(R, Z, k + 1) # Get the magnetic field components
    # Add up the squared norm of the B-field components and take the logarithm of it
    log_Bn2[k, :, :] = log10((BnR * conj(BnR) +
                              Bnphi * conj(Bnphi) +
                              BnZ * conj(BnZ)).real)
# color plot normalization 
norm = Normalize(vmin=amin(log_Bn2), vmax=amax(log_Bn2))
#%% 

# Create a figure with subplots for each mode
fig = plt.figure(layout='constrained')
axs = fig.subplots(vert, horz).ravel()
for k in range(vert * horz):
    # Create colorplot
    im = axs[k].imshow(log_Bn2[k, :, :].T, origin='lower', cmap='magma',
                       extent=[R_min, R_max, Z_min, Z_max])
    im.set_norm(norm) # for colors of the image
    axs[k].set_title(f"$n = {k + 1}$")

# Add colorbar and labels
cbar = fig.colorbar(im, ax=axs, location='bottom')
cbar.set_label(r'$\log_{10} |\vec{B}_{n}|^{2}$')
plt.show()