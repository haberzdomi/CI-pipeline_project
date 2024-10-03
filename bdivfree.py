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
    global icall_c, AnR_Re, AnR_Im, AnZ_Re, AnZ_Im
    if icall_c == 0:
        icall_c = 1
        BR, Bphi, BZ = read_field4()
        AnR_Re, AnR_Im, AnZ_Re, AnZ_Im = vector_potentials(64, BR, Bphi, BZ)
    BnR, Bnphi, BnZ = field_divfree(R, Z, n, AnR_Re, AnR_Im, AnZ_Re, AnZ_Im)
    return BnR, Bnphi, BnZ

def read_field4():
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
    for n in range(1, n_max + 1):
        fourier = exp(-1j * n * linspace(0, 2 * pi, nphi)) / (nphi - 1);
        for kR in range(nR):
            for kZ in range(nZ):
                AnR[kR, kZ] = 1j / n * R_eqd[kR] * sum(BnZ[kR, :-1, kZ] * fourier[:-1])
                AnZ[kR, kZ] = -1j / n * R_eqd[kR] * sum(BnR[kR, :-1, kZ] * fourier[:-1])
        AnR_Re[n-1] = RectBivariateSpline(R_eqd, Z_eqd, AnR.real, kx=5, ky=5)
        AnR_Im[n-1] = RectBivariateSpline(R_eqd, Z_eqd, AnR.imag, kx=5, ky=5)
        AnZ_Re[n-1] = RectBivariateSpline(R_eqd, Z_eqd, AnZ.real, kx=5, ky=5)
        AnZ_Im[n-1] = RectBivariateSpline(R_eqd, Z_eqd, AnZ.imag, kx=5, ky=5)
    return AnR_Re, AnR_Im, AnZ_Re, AnZ_Im

def field_divfree(R, Z, n, AnR_Re, AnR_Im, AnZ_Re, AnZ_Im):
    global nR, nphi, nZ, R_min, R_max, phi_min, phi_max, Z_min, Z_max
    from numpy import atleast_1d, newaxis, squeeze
    R = atleast_1d(R)
    Z = atleast_1d(Z)
    AnR = AnR_Re[n-1](R, Z) + 1j * AnR_Im[n-1](R, Z)
    AnZ = AnZ_Re[n-1](R, Z) + 1j * AnZ_Im[n-1](R, Z)
    dAnR_dZ = AnR_Re[n-1](R, Z, dy=1) + 1j * AnR_Im[n-1](R, Z, dy=1)
    dAnZ_dR = AnZ_Re[n-1](R, Z, dx=1) + 1j * AnZ_Im[n-1](R, Z, dx=1)
    BnR = 1j * n * AnZ / R[:, newaxis]
    Bnphi = dAnR_dZ - dAnZ_dR
    BnZ = -1j * n * AnR / R[:, newaxis]
    return squeeze(BnR), squeeze(Bnphi), squeeze(BnZ)