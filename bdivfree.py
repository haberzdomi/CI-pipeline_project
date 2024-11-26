from numpy import atleast_1d, empty, exp, linspace, newaxis, pi, squeeze, sum
from scipy.interpolate import RectBivariateSpline


class A_splines:
    """Class to store the spline functions for the vector potential components."""

    def __init__(self, n=0):
        self.AnR_Re = [None] * n
        self.AnR_Im = [None] * n
        self.AnZ_Re = [None] * n
        self.AnZ_Im = [None] * n


def vector_potentials(n_max, grid, BnR, Bnphi, BnZ):
    """Calculate the first n_max modes of the R- and Z-components of the vector potential using fourier transformation and B=rot(A).
    Perform a bivariate spline approximation to evaluate them on the descretized k-space points.

    Args:
        n_max (int): highest mode number to be calculated
        grid (grid_parameters): Object containing the cylindrical 3D-grid and its parameters.
        BnR (array[float], shape=(nR, nphi, nZ)): R-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
        Bnphi (array[float], shape=(nR, nphi, nZ)): phi-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
        BnZ (array[float], shape=(nR, nphi, nZ)): Z-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
    Returns:
        A (A_splines): Object containing the spline functions for the first n modes of the real and imaginary parts of the radial and axial vector potential components.

    """
    A = A_splines(n_max)

    AnR = empty((grid.nR, grid.nZ), dtype=complex)
    AnZ = empty((grid.nR, grid.nZ), dtype=complex)

    # Get first n_max modes of the vector potential
    for n in range(1, n_max + 1):
        fourier_coef = exp(-1j * n * linspace(0, 2 * pi, grid.nphi)) / (grid.nphi - 1)

        # Calculate the n'th mode of the radial and axial components of
        # the vector potential using fourier transformation and B=rot(A).
        for kR in range(grid.nR):
            for kZ in range(grid.nZ):
                AnR[kR, kZ] = (
                    1j / n * grid.R[kR] * sum(BnZ[kR, :-1, kZ] * fourier_coef[:-1])
                )
                AnZ[kR, kZ] = (
                    -1j / n * grid.R[kR] * sum(BnR[kR, :-1, kZ] * fourier_coef[:-1])
                )
        # Interpolation via bivariate spline approximation. The data values (3rd arg) is
        # defined on the grid (1st and 2nd arg). kx, ky are the degrees of the bivariate spline.
        A.AnR_Re[n - 1] = RectBivariateSpline(grid.R, grid.Z, AnR.real, kx=5, ky=5)
        A.AnR_Im[n - 1] = RectBivariateSpline(grid.R, grid.Z, AnR.imag, kx=5, ky=5)
        A.AnZ_Re[n - 1] = RectBivariateSpline(grid.R, grid.Z, AnZ.real, kx=5, ky=5)
        A.AnZ_Im[n - 1] = RectBivariateSpline(grid.R, grid.Z, AnZ.imag, kx=5, ky=5)
    return A


def field_divfree(R, Z, n, A):
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
    # Calculate the azimuthal component such that div(B) = 0
    BnR = 1j * n * AnZ / R[:, newaxis]
    Bnphi = dAnR_dZ - dAnZ_dR
    BnZ = -1j * n * AnR / R[:, newaxis]
    # Remove 1D axis before returning
    return squeeze(BnR), squeeze(Bnphi), squeeze(BnZ)
