#%%
icall_c = 0
AnR_Re = None
AnR_Im = None
AnZ_Re = None
AnZ_Im = None
nR = 0
nphi = 0
nZ = 0
R_min = 0.0
R_max = 0.0
phi_min = 0.0
phi_max = 0.0
Z_min = 0.0
Z_max = 0.0

def field_c(R, Z, n):
    """Evaluate the n'th mode of the magnetic field at the given discretized space points.

    Args:
        R (array[float], shape=(2*nR-1,)): R-coordinates of the 2*nR-1 grid points for evaluation of the magnetic field
        Z (array[float], shape=(2*nZ-1,)): Z-coordinates of the 2*nR-1 grid points for evaluation of the magnetic field
        n (int): mode number

    Returns:
        BnR (array[complex float], shape=(len(R),len(Z))): R-component of the n'th mode of the magnetic field
        Bnphi (array[complex float], shape=(len(R),len(Z))): phi-component of the n'th mode of the magnetic field
        BnZ (array[complex float], shape=(len(R),len(Z))): Z-component of the n'th mode of the magnetic field
    """
    global icall_c, AnR_Re, AnR_Im, AnZ_Re, AnZ_Im
    # Read the magnetic field components from the output file and calculate the vector potentials.
    # If icall_c is 1, they are already determined and stored in the global variables.
    if icall_c == 0:
        icall_c = 1
        BR, Bphi, BZ = read_field4()
        AnR_Re, AnR_Im, AnZ_Re, AnZ_Im = vector_potentials(64, BR, Bphi, BZ)
    BnR, Bnphi, BnZ = field_divfree(R, Z, n, AnR_Re, AnR_Im, AnZ_Re, AnZ_Im)
    return BnR, Bnphi, BnZ

def read_field4():
    """Get the magnetic field components for each point in the discretized 3D space from the output field.dat file.

    Returns:
        BR (array[float], shape=(nR, nphi, nZ)): R-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
        Bphi (array[float], shape=(nR, nphi, nZ)): phi-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
        BZ (array[float], shape=(nR, nphi, nZ)): Z-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
    """
    global nR, nphi, nZ, R_min, R_max, phi_min, phi_max, Z_min, Z_max
    from numpy import empty
    f = open('field.dat', 'r')
    nR, nphi, nZ, dummy = [int(data) for data in f.readline().split()]
    R_min, R_max = [float(data) for data in f.readline().split()]
    phi_min, phi_max = [float(data) for data in f.readline().split()]
    Z_min, Z_max = [float(data) for data in f.readline().split()]
    BR = empty((nR, nphi, nZ))
    Bphi = empty((nR, nphi, nZ))
    BZ = empty((nR, nphi, nZ))
    for kR in range(nR):
        for kphi in range(nphi):
            for kZ in range(nZ):
                BR[kR, kphi, kZ], Bphi[kR, kphi, kZ], BZ[kR, kphi, kZ] = [float(data) for data in f.readline().split()]
    f.close()
    return BR, Bphi, BZ

def vector_potentials(n_max, BnR, Bnphi, BnZ):
    """ Calculate the first n_max modes of the R- and Z-components of the vector potential using fourier transformation and B=rot(A). 
    Perform a bivariate spline approximation to evaluate them on the descretized k-space points.

    Args:
        n_max (int): highest mode number to be calculated
        BnR (array[float], shape=(nR, nphi, nZ)): R-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
        Bnphi (array[float], shape=(nR, nphi, nZ)): phi-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
        BnZ (array[float], shape=(nR, nphi, nZ)): Z-component of the magnetic field for the calculated (nR, nphi, nZ)-grid points.
    Returns:
        AnR_Re (List[RectBivariateSpline], size=n_max): Real part of the first n modes of the radial component of the vector potential.
        AnR_Im (List[RectBivariateSpline], size=n_max): Imaginary part of the first n modes of the radial component of the vector potential.
        AnZ_Re (List[RectBivariateSpline], size=n_max): Real part of the first n modes of the axial component of the vector potential.
        AnZ_Im (List[RectBivariateSpline], size=n_max): Imaginary part of the first n modes of the axial component of the vector potential.
    """
    global nR, nphi, nZ, R_min, R_max, phi_min, phi_max, Z_min, Z_max
    from numpy import empty, exp, linspace, pi, sum
    from scipy.interpolate import RectBivariateSpline
    AnR_Re = [None] * n_max
    AnR_Im = [None] * n_max
    AnZ_Re = [None] * n_max
    AnZ_Im = [None] * n_max
    R_eqd = linspace(R_min, R_max, nR)
    Z_eqd = linspace(Z_min, Z_max, nZ)
    AnR = empty((nR, nZ), dtype=complex)
    AnZ = empty((nR, nZ), dtype=complex)
    
    # Get first n_max modes of the vector potential
    for n in range(1, n_max + 1):
        fourier = exp(-1j * n * linspace(0, 2 * pi, nphi)) / (nphi - 1); # Fourier coefficients
        # Calculate the n'th mode of the radial and axial components of 
        # the vector potential using fourier transformation and B=rot(A).
        for kR in range(nR):
            for kZ in range(nZ):
                AnR[kR, kZ] = 1j / n * R_eqd[kR] * sum(BnZ[kR, :-1, kZ] * fourier[:-1])
                AnZ[kR, kZ] = -1j / n * R_eqd[kR] * sum(BnR[kR, :-1, kZ] * fourier[:-1])
        # Interpolation via bivariate spline approximation. 
        # First two args are the points for which the data is evaluated.
        # The third argument is the data. 
        # The last two arguments are the degree of the bivariate spline.
        AnR_Re[n-1] = RectBivariateSpline(R_eqd, Z_eqd, AnR.real, kx=5, ky=5)
        AnR_Im[n-1] = RectBivariateSpline(R_eqd, Z_eqd, AnR.imag, kx=5, ky=5)
        AnZ_Re[n-1] = RectBivariateSpline(R_eqd, Z_eqd, AnZ.real, kx=5, ky=5)
        AnZ_Im[n-1] = RectBivariateSpline(R_eqd, Z_eqd, AnZ.imag, kx=5, ky=5)
    return AnR_Re, AnR_Im, AnZ_Re, AnZ_Im

def field_divfree(R, Z, n, AnR_Re, AnR_Im, AnZ_Re, AnZ_Im):
    """Calculate the n'mode of the magnetic field from the n'th mode of the vector potential 
    components and make it divergence-free by calculating the azimuthal component appropriately.

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
    global nR, nphi, nZ, R_min, R_max, phi_min, phi_max, Z_min, Z_max
    from numpy import atleast_1d, newaxis, squeeze
    # Make sure R and Z are arrays
    R = atleast_1d(R) 
    Z = atleast_1d(Z)
    # Get the R- and Z-components of the n'th mode of the vector potential and their derivatives
    AnR = AnR_Re[n-1](R, Z) + 1j * AnR_Im[n-1](R, Z)
    AnZ = AnZ_Re[n-1](R, Z) + 1j * AnZ_Im[n-1](R, Z)
    dAnR_dZ = AnR_Re[n-1](R, Z, dy=1) + 1j * AnR_Im[n-1](R, Z, dy=1)
    dAnZ_dR = AnZ_Re[n-1](R, Z, dx=1) + 1j * AnZ_Im[n-1](R, Z, dx=1)
    # Calculate the azimuthal component such that div(B) = 0
    BnR = 1j * n * AnZ / R[:, newaxis]
    Bnphi = dAnR_dZ - dAnZ_dR
    BnZ = -1j * n * AnR / R[:, newaxis]
    # Remove 1D axis before returning
    return squeeze(BnR), squeeze(Bnphi), squeeze(BnZ)