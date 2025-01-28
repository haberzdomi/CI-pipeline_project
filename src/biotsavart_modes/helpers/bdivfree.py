import numpy as np
from scipy.interpolate import RectBivariateSpline


class A_SPLINES:
    def __init__(self, n=0):
        """Class to store the spline functions for the vector potential components.

        Args:
            n (int, optional): highest mode number to be calculated. Defaults to 0.

        Attributes:
            AnR_Re (List[RectBivariateSpline], size=n): Spline function of the real part of the first n modes of the radial component of the vector potential.
            AnR_Im (List[RectBivariateSpline], size=n): Spline function of the imaginary part of the first n modes of the radial component of the vector potential.
            AnZ_Re (List[RectBivariateSpline], size=n): Spline function of the real part of the first n modes of the axial component of the vector potential.
            AnZ_Im (List[RectBivariateSpline], size=n): Spline function of the imaginary part of the first n modes of the axial component of the vector potential.
            Defaults to None lists of size n.
        """

        self.AnR_Re = [None] * n
        self.AnR_Im = [None] * n
        self.AnZ_Re = [None] * n
        self.AnZ_Im = [None] * n


def get_A_field_modes(grid, BR, Bphi, BZ):
    """Calculate the first n_max modes of the R- and Z-components of the vector potential using fourier transformation and B=rot(A).
    Perform a bivariate spline approximation to evaluate them on the descretized k-space points.

    Args:
        n_max (int): highest mode number to be calculated
        grid (GRID object): Object containing the cylindrical 3D-grid and its parameters.
        BR (array[float], shape=(nR, nphi, nZ)): R-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
        Bphi (array[float], shape=(nR, nphi, nZ)): phi-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
        BZ (array[float], shape=(nR, nphi, nZ)): Z-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
    Returns:
        A (A_SPLINES): Object containing the spline functions for the first n modes of the real and imaginary parts of the radial and axial vector potential components.
    """

    # Get the maximum mode number based on Nyquist theorem. The minus one is
    # because the last data point in phi-direction is the same as the first one.
    n_max = int((grid.nphi - 1) / 2)

    # A_factor comes from B = rot(A).

    A_factor = (1j / np.arange(1, n_max + 1)[:, np.newaxis] * grid.R)[:, :, np.newaxis]

    # Get first n_max modes of the radial component of the vector potential.
    BnR = np.fft.fft(BZ[:, :-1, :], axis=1, norm="forward")
    BnR = np.swapaxes(BnR, 0, 1)[1 : n_max + 1]
    AnR = A_factor * BnR

    # Get first n_max modes of the axial component of the vector potential.
    BnZ = np.fft.fft(BR[:, :-1, :], axis=1, norm="forward")
    BnZ = np.swapaxes(BnZ, 0, 1)[1 : n_max + 1]
    AnZ = -A_factor * BnZ

    # Interpolation via bivariate spline approximation.
    A = A_SPLINES(n_max)
    for k in range(n_max):
        # The data values (3rd arg) are defined on the grid (1st and 2nd arg).
        # kx, ky are the degrees of the bivariate spline.
        A.AnR_Re[k] = RectBivariateSpline(grid.R, grid.Z, AnR[k].real, kx=5, ky=5)
        A.AnR_Im[k] = RectBivariateSpline(grid.R, grid.Z, AnR[k].imag, kx=5, ky=5)
        A.AnZ_Re[k] = RectBivariateSpline(grid.R, grid.Z, AnZ[k].real, kx=5, ky=5)
        A.AnZ_Im[k] = RectBivariateSpline(grid.R, grid.Z, AnZ[k].imag, kx=5, ky=5)
    return A


def calc_B_field_modes(R, Z, n, A):
    """Calculate the n'mode of the magnetic field from the n'th mode of the vector potential
    components with the assumption, that the phi-component of the vector potential is zero, in order to make it divergence-free

    Args:
        R (array[float], shape=(2*nR-1,)): R-coordinates of the 2*nR-1 grid points for evaluation of the magnetic field
        Z (array[float], shape=(2*nZ-1,)): Z-coordinates of the 2*nR-1 grid points for evaluation of the magnetic field
        n (int): mode number
        AnR_Re (List[RectBivariateSpline], size=n_max): Real part of the first n modes of the radial component of the vector potential.
        AnR_Im (List[RectBivariateSpline], size=n_max): Imaginary part of the first n modes of the radial component of the vector potential.
        AnZ_Re (List[RectBivariateSpline], size=n_max): Real part of the first n modes of the axial component of the vector potential.
        AnZ_Im (List[RectBivariateSpline], size=n_max): Imaginary part of the first n modes of the axial component of the vector potential.

    Returns:
        BnR (array[complex float], shape=(len(R),len(Z))): R-component of the n'th mode of the magnetic field
        Bnphi (array[complex float], shape=(len(R),len(Z))): phi-component of the n'th mode of the magnetic field
        BnZ (array[complex float], shape=(len(R),len(Z))): Z-component of the n'th mode of the magnetic field
    """
    # Get the R- and Z-components of the n'th mode of the vector potential and their derivatives
    AnR = A.AnR_Re[n - 1](R, Z) + 1j * A.AnR_Im[n - 1](R, Z)
    AnZ = A.AnZ_Re[n - 1](R, Z) + 1j * A.AnZ_Im[n - 1](R, Z)
    dAnR_dZ = A.AnR_Re[n - 1](R, Z, dy=1) + 1j * A.AnR_Im[n - 1](R, Z, dy=1)
    dAnZ_dR = A.AnZ_Re[n - 1](R, Z, dx=1) + 1j * A.AnZ_Im[n - 1](R, Z, dx=1)

    # Get the magnetic field components from the vector potential components
    BnR = 1j * n * AnZ / R[:, np.newaxis]
    Bnphi = dAnR_dZ - dAnZ_dR  # Calculate the azimuthal component such that div(B) = 0
    BnZ = -1j * n * AnR / R[:, np.newaxis]
    # Remove 1D axis before returning
    return np.squeeze(BnR), np.squeeze(Bnphi), np.squeeze(BnZ)
