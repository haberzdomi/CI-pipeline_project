import math

class grid:
    def __init__(self,n_R,n_phi,n_Z,R_min,R_max,phi_min,phi_max,Z_min,Z_max):
        self.n_R=n_R
        self.n_phi=n_phi
        self.n_Z=n_Z
        self.R_min=R_min
        self.R_max=R_max
        self.phi_min=phi_min
        self.phi_max=phi_max
        self.Z_min=Z_min
        self.Z_max=Z_max
        hr = (R_max - R_min)/(n_R-1) # step size in r direction
        hphi = (phi_max - phi_min)/(n_phi-1) # step size in phi direction
        hz = (Z_max - Z_min)/(n_Z-1)
        self.R=[R_min+hr*i for i in range(n_R)]
        self.phi=[phi_min+hphi*i for i in range(n_phi)]
        self.Z=[Z_min+hz*i for i in range(n_Z)]
