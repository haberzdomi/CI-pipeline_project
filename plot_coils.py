#%%
from numpy import empty, loadtxt
import matplotlib.pyplot as plt

data = loadtxt('co_asd.dd', skiprows=1, usecols=(0, 1, 2))
ncoil = 16
nseg = data.shape[0] // ncoil
X = data[:, 0].reshape((ncoil, nseg))
Y = data[:, 1].reshape((ncoil, nseg))
Z = data[:, 2].reshape((ncoil, nseg))

#%%
ax = plt.figure().add_subplot(projection='3d')
for k in range(ncoil):
  ax.plot(X[k, :], Y[k, :], Z[k, :], '-k')
plt.show()
