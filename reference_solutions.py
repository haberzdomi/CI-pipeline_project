import math
from biotsavart import make_field_file_from_coils
from plot_modes import read_field
import os
import numpy as np
from bdivfree import get_A_field_modes, calc_B_field_modes

def BZ_formula(z, R, I):
    """Analytical formula for the magnetic field of a circular current loop along its axis

    Args:
        z (float): position along the axis
        R (float): Radius of the loop
        I (float): current flowing through the loop
    
    Returns:
        Result of the formula
    """
    return I*R**2*2*math.pi/(R**2+z**2)**(3/2)



def circular_current(R_max, nR, nphi, nZ, R_0, I_c, nseg):
    """Use biotsavart.py to calculate the magnetic field of a circular current loop

    Args:
        R_max (float): the extent of the computational grid, i.e., R ∈ [0,Rmax], Z ∈ [-Rmax,Rmax]
        nR (int), nphi (int), nZ (int): the number of grid points
        R_0 (float): Radius of the loop
        I_c (float): current flowing through the loop
        nseg (int): the number of segments in the discretisation of the loop
    
    Returns:
        Z (array[float], shape=(nZ,)): the array of equidistant Z coordinates of the computational grid
        BZ (array[float], shape=(nZ,)): the values of BZ(Z), as returned by biotsavart_asdex
        BZ_analytical (array[float], shape=(nZ,)): the values of BZ(Z), as returned by the analytical formula
    """
    #Create input files for test
    file1 = open("grid.inp", "w")
    print(nR, nphi, nZ, file=file1)
    print(0, R_max, file=file1)
    print(-R_max, R_max, file=file1)
    file1.close()
    file2 = open("coil.dd", "w")
    print(nseg+1, file=file2)
    dphi = 2*math.pi/nseg
    for i in range(nseg):
        print(R_0*math.cos(i*dphi), R_0*math.sin(i*dphi), 0, 1, 1, file=file2)
    print(R_0, 0, 0, 0, 1, file=file2)  #connect last point to first point
    file2.close()
    file3 = open("current.dd", "w")
    print(I_c, file=file3)
    file3.close()


    #run biotsavart_asdex
    make_field_file_from_coils('grid.inp', 'coil.dd', 'current.dd', 'output.dat', 1)

    #process output data
    data = open("output.dat", 'r')
    for i in range(4):
        data.readline()
    BZ_data = np.empty((nR, nphi, nZ))
    for kR in range(nR):
        for kphi in range(nphi):
            for kZ in range(nZ):
                BZ_data[kR, kphi, kZ] = float(data.readline().split()[2])
    data.close()
    
    BZ = [BZ_data[0,0,i] for i in range(nZ)]
    Z = np.linspace(-R_max, R_max, nZ)
    BZ_analytic = [BZ_formula(z, R_0, I_c) for z in Z]
    

    #remove test input files
    for filename in ['grid.inp', 'coil.dd', 'current.dd', 'output.dat']:
        os.remove(filename)

    return Z, BZ, BZ_analytic



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
    # Get grid and magnetic field components from the calculation output field.dat
    g, BR, Bphi, BZ = read_field('field.dat')
    
    # Calculate the first 'n_max' modes of the magnetic field on the radial grid ...
    
    # ... via the handwritten fourier transform in field_c
    A = get_A_field_modes(g, BR, Bphi, BZ)
    BnR=np.empty((n_max, g.nR), dtype=complex)
    for k in range(n_max):
        # n=k+1 because range starts from 0 but n=1 is the first mode.
        # Fix Z to the middle of the axial grid
        BnR[k], Bnphi, BnZ = calc_B_field_modes(g.R, g.Z[g.nZ//2], k + 1, A)    
    
    # ... via the method numpy.fft.fft. 
    BnR_fft=np.empty((n_max, g.nR))
    # The fourier transform implemented in field_c is "forward".
    # phi goes from 0 to 2*pi therefore the end is excluded.
    # numpy stores the components such that the first mode is on index 1 and the higher order modes follow on the next indices.
    BnR_fft = np.fft.fft(BR[:,:-1, g.nZ//2],norm="forward")[:,1:n_max+1]
    BnR_fft = BnR_fft.T # Change to the same shape as BnR.
    return g.R, BnR, BnR_fft