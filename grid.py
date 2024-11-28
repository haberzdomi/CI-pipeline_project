from numpy import linspace


class grid:
    def __init__(self, nR, nphi, nZ, R_min, R_max, phi_min, phi_max, Z_min, Z_max):
        """Object to store the parameters for the discretized grid and the grid itself.

        Attributes:
            nR (int): Number of points in the radial coordinate.
            nphi (int): Number of points in the azimuthal coordinate.
            nZ (int): Number of points in the axial coordinate.
            R_min (float): Minimum value of the radial coordinate.
            R_max (float): Maximum value of the radial coordinate.
            phi_min (float): Minimum value of the azimuthal coordinate.
            phi_max (float): Maximum value of the azimuthal coordinate.
            Z_min (float): Minimum value of the axial coordinate.
            Z_max (float): Maximum value of the axial coordinate.
            R (array[float], shape=(n_R, )): Grid points in the radial direction.
            phi (array[float], shape=(n_phi, )): Grid points in the azimuthal direction.
            Z (array[float], shape=(n_Z, )): Grid points in the axial direction.

        """
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
