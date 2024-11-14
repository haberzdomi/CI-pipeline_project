# Dominik Haberz 11916636
# Clara Rinner 01137166
#%%
# program biotsavart
"""
Calculate the magnetic field components on a grid defined in the 
file 'biotsavart.inp' using the Biot-Savart law. The output is 
written to the file 'field.dat'.
"""
#
from vvn_asdex import calculate_magnetic_field, load_coil_data
#%%
import math
def biotsavart_asdex():
    coil_data = load_coil_data()
    x = [0.0, 0.0, 0.0]
    #
    # Get the input data, defining the discrete space for the magnetic field calculation.
    f1=open('biotsavart.inp','r')
    nr,np,nz=[int(data) for data in f1.readline().split()] # number of grid points in r, phi, z direction
    rmin, rmax=[float(data) for data in f1.readline().split()] # min and max values for r
    zmin, zmax=[float(data) for data in f1.readline().split()] # min and max values for z
    f1.close()
    #
    L1i=1
    #
    pmin=0.0
    pmax=2*math.pi/L1i
    #
    hrad = (rmax - rmin)/(nr-1) # step size in r direction
    hphi = (pmax - pmin)/(np-1) # step size in phi direction
    hzet = (zmax - zmin)/(nz-1) # step size in z direction
    #
    #%%
    f1=open('field.dat','w')
    # Write the input parameters for the magnetic field calculation to the output file.
    print(nr,np,nz,L1i,file=f1)
    print(rmin,rmax,file=f1)
    print(pmin,pmax,file=f1)
    print(zmin,zmax,file=f1)
    # Loop over the grid points and calculate the magnetic field components.
    for i in range(nr):
        print(i+1,'/',nr) # print progress
        # TODO: Dynamic creation of matrices is more time consuming (especailly for higher dimensions) than preallocating them
        for j in range(np):
            for k in range(nz):
                x = [rmin+hrad*i,pmin+hphi*j,zmin+hzet*k] # coordinates for current grid point
                b=calculate_magnetic_field(x, coil_data)
                print(b[0],b[1],b[2],file=f1)
    f1.close()
    #
