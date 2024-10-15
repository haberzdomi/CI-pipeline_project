#%%
from numpy import amin, amax, conj, empty, linspace, log10
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from bdivfree import field_c

"""
Plots the decadic logarithm of the square of the norm 
of the total magnetic field in each point in discretized space. 
There is one colorplot for each of the n coils.
"""

#%%
with open('biotsavart.inp', 'r') as f:
    nR, nphi, nZ = [int(data) for data in f.readline().split()]  # number of points for each dimension for discretization
    R_min, R_max = [float(data) for data in f.readline().split()]
    Z_min, Z_max = [float(data) for data in f.readline().split()]

# double the resolution
R = linspace(R_min, R_max, 2 * nR - 1)
Z = linspace(Z_min, Z_max, 2 * nZ - 1)

vert = 2
horz = 4
log_Bn2 = empty((vert * horz, 2 * nR - 1, 2 * nZ - 1))
for k in range(vert * horz):
    BnR, Bnphi, BnZ = field_c(R, Z, k + 1)
    # Add up the squared norm of the B-field components
    log_Bn2[k, :, :] = log10((BnR * conj(BnR) +
                              Bnphi * conj(Bnphi) +
                              BnZ * conj(BnZ)).real)
# color plot normalization 
norm = Normalize(vmin=amin(log_Bn2), vmax=amax(log_Bn2))
#%%
fig = plt.figure(layout='constrained')
axs = fig.subplots(vert, horz).ravel()
for k in range(vert * horz):
    im = axs[k].imshow(log_Bn2[k, :, :].T, origin='lower', cmap='magma',
                       extent=[R_min, R_max, Z_min, Z_max])
    im.set_norm(norm)
    axs[k].set_title(f"$n = {k + 1}$")
cbar = fig.colorbar(im, ax=axs, location='bottom')
cbar.set_label(r'$\log_{10} |\vec{B}_{n}|^{2}$')
plt.show()