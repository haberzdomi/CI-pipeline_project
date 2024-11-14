import math
from biotsavart_asdex import biotsavart_asdex
import os
import numpy as np
import bdivfree

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
    files_to_move = ["biotsavart.inp", "co_asd.dd", "cur_asd.dd"]
    #move original input files to temporary folder
    os.mkdir("C:\\Users\\franz\\OneDrive\\Uni\\Software Engineering in Physics\\worstpractice\\temporary")
    for filename in files_to_move:
        os.rename("C:\\Users\\franz\\OneDrive\\Uni\\Software Engineering in Physics\\worstpractice\\"+filename, "C:\\Users\\franz\\OneDrive\\Uni\\Software Engineering in Physics\\worstpractice\\temporary\\"+filename)
    
    #Create input files for test
    file1 = open("biotsavart.inp", "w")
    print(nR, nphi, nZ, file=file1)
    print(0, R_max, file=file1)
    print(-R_max, R_max, file=file1)
    file1.close()
    file2 = open("co_asd.dd", "w")
    print(nseg+1, file=file2)
    dphi = 2*math.pi/nseg
    for i in range(nseg):
        print(R_0*math.cos(i*dphi), R_0*math.sin(i*dphi), 0, 1, 1, file=file2)
    print(R_0, 0, 0, 0, 1, file=file2)  #connect last point to first point
    file2.close()
    file3 = open("cur_asd.dd", "w")
    print(I_c, file=file3)
    file3.close()

    #
    import vvn_asdex
    vvn_asdex.PROP = False

    #run biotsavart_asdex
    biotsavart_asdex()

    #process output data
    data = open("field.dat", 'r')
    for i in range(4):
        data.readline()
    BZ_data = np.empty((nR, nphi, nZ))
    for kR in range(nR):
        for kphi in range(nphi):
            for kZ in range(nZ):
                BZ_data[kR, kphi, kZ] = float(data.readline().split()[2])
    
    BZ = [BZ_data[0,0,i] for i in range(nZ)]
    Z = np.linspace(-R_max, R_max, nZ)
    BZ_analytic = [BZ_formula(z, R_0, I_c) for z in Z]
    

    #remove test input files
    for filename in files_to_move:
        os.remove(filename)

    #return original input files to their former place
    for filename in files_to_move:
        os.rename("C:\\Users\\franz\\OneDrive\\Uni\\Software Engineering in Physics\\worstpractice\\temporary\\"+filename, "C:\\Users\\franz\\OneDrive\\Uni\\Software Engineering in Physics\\worstpractice\\"+filename)
    os.rmdir("temporary")
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