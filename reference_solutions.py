import numpy as np
import bdivfree

def fourier_analysis(n_max):
    """ Calculate the first 'n_max' modes of the magnetic field on the radial grid R via handwritten discrete fourier transform 
    using field_c and via numpy.fft.fft. The input parameters and magnetic field components are read from field.dat.

    Args:
        n_max (int): Highest mode number up to which the of the magnetic field is calculated.

    Returns:
        R (array[float], shape=(nR,)): Radial grid for which the magentic field modes are calculated.
        BnR (array[complex float], shape=(n_max,nR)): R-component of the n'th mode of the magnetic field calculated
                                                      by field_c using a handwritten fourier transform.
        BnR_fft (array[complex float], shape=(n_max,nR)): R-component of the n'th mode of the magnetic field calculated
                                                      by numpy.fft.fft using fast fourier transform.
    """
    
    # Get input parameters for the radial and axial grid.
    with open('field.dat', 'r') as f:
        # number of grid points for each dimension for discretization
        nR, nphi, nZ, L1i = [int(data) for data in f.readline().split()]
        # Boundaries of R- and Z-dimension
        R_min, R_max = [float(data) for data in f.readline().split()]
        phi_min, phi_max = [float(data) for data in f.readline().split()]
        Z_min, Z_max = [float(data) for data in f.readline().split()]
    
    # Create grids
    R = np.linspace(R_min, R_max, nR)
    Z = np.linspace(Z_min, Z_max, nZ)
    
    # Fix Z to the middle of the axial grid
    Z_fixed = Z[nZ//2]
    
    # Calculate the first 'n_max' modes of the magnetic field on the radial grid ...
    
    # ... via the handwritten fourier transform in field_c
    BnR=np.empty((n_max, nR), dtype=complex)
    for k in range(n_max):
        # n=k+1 because range starts from 0 but n=1 is the first mode.
        BnR[k], Bnphi, BnZ = bdivfree.field_c(R, Z_fixed, k + 1)    
    
    # ... via the method numpy.fft.fft. 
    BR, Bphi, BZ = bdivfree.read_field4() # Get magnetic field components from field.dat
    BnR_fft=np.empty((n_max, nR))
    # The fourier transform implemented in field_c is "forward".
    # phi goes from 0 to 2*pi therefore the end is excluded.
    # numpy stores the components such that the first mode is on index 1 and the higher order modes follow on the next indices.
    BnR_fft = np.fft.fft(BR[:,:-1,nZ//2],norm="forward")[:,1:n_max+1]
    BnR_fft = BnR_fft.T # Change to the same shape as BnR.
    return R, BnR, BnR_fft