from numpy import linspace


class grid:
    def __init__(self, nR, nphi, nZ, R_min, R_max, phi_min, phi_max, Z_min, Z_max):
        self.nR = nR
        self.nphi = nphi
        self.nZ = nZ
        self.R_min = R_min
        self.R_max = R_max
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.Z_min = Z_min
        self.Z_max = Z_max
        self.R = linspace(R_min, R_max, nR)
        self.phi = linspace(phi_min, phi_max, nphi)
        self.Z = linspace(Z_min, Z_max, nZ)
