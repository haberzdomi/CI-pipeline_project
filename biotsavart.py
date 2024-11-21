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
import numpy as np
import math
from grid import make_grid

class coils:
    def __init__(self,X,Y,Z,has_current,coil_number,n_nodes):
        self.X=X
        self.Y=Y
        self.Z=Z
        self.has_current=has_current
        self.coil_number=coil_number
        self.n_nodes=n_nodes

def calc_biotsavart(grid_coordinates, coil_data, currents):
    """
    Calculate the magnetic field components by 
    evaluating the Biot-Savart integral.

    Args:
        grid_coordinates (array[float], shape=(3)): cylindrical coordinates of the grid point
        coil_data: coordinates of the coil points

    Returns:
        BRI (float): radial component fo the magnetic field
        BfI (float): toroidal component of the magnetic field
        BZI (float): axial component of the magnetic field
    """

    RI=grid_coordinates[0]
    fI=grid_coordinates[1]
    ZI=grid_coordinates[2]
    cosf=np.cos(fI)
    sinf=np.sin(fI)
    Y_grid=RI*sinf  #cartesian coordinates of grid point
    X_grid=RI*cosf
    Z_grid=ZI
    grid_point=[X_grid,Y_grid,Z_grid]

    B=[0,0,0]

    coil_point_current=[coil_data.X[0],coil_data.Y[0],coil_data.Z[0]]
    R1_vector=np.subtract(grid_point,coil_point_current)  #difference of grid point and first coil point | r-r'[0]

    R1=np.linalg.norm(R1_vector)  #distance between grid point and first coil point | |r-r'[0]|

    for K in range(1,coil_data.n_nodes):   # loops through the remaining coil points

        coil_point_previous=coil_point_current
        coil_point_current=[coil_data.X[K],coil_data.Y[K],coil_data.Z[K]]

        A=np.subtract(coil_point_current, coil_point_previous)  #difference between the current coil point and the previous one | l

        R2_vector=np.subtract(grid_point,coil_point_current)#difference between grid point and current coil point | r-r'[k]

        scalar_product = np.dot(A, R2_vector)   #scalar product of l and r-r'[k]
        R2=np.linalg.norm(R2_vector)  #distance between grid point and current coil point | |r-r'[k]|


        if coil_data.has_current[K-1]!=0.0:   # if not first point of a coil

            OBCP=1.0/(R2*(R1+R2)+scalar_product)
            FAZRDA=-(R1+R2)*OBCP/R1/R2*currents[coil_data.coil_number[K]-1] #curco[nco[K]-1] - current in coil
            B_B1=np.cross(R2_vector,A)    #r-r'[k] x l

            B=np.add(np.dot(B_B1, FAZRDA), B)

        R1=R2

    B_R=B[0]*cosf+B[1]*sinf
    B_phi=B[1]*cosf-B[0]*sinf
    B_Z = B[2]

    return B_R, B_phi, B_Z


def read_coils(coil_file):
    """
    Return the data in co_asd.dd and cur_asd.dd
    
    Args: 
        None
    Returns:
        XO, YO, ZO - coodinate lists of coil points
        cur - 0 if last point in a coil, otherwise 1
        nco - coil number
        nnodc - number of coil points
    """

    file = open(coil_file, 'r')
    n_nodes = int(file.readline())

        
    data = np.loadtxt(file)
    X = data[:,0]
    Y = data[:,1]
    Z = data[:,2]
    has_current = data[:,3]
    coil_number = [int(d) for d in data[:,4]]


    # Raise error if the current factor for the last node of the last coil is not 0.
    if has_current[n_nodes-1]!=0.0:
        raise Exception(str(has_current[n_nodes-1])+'=cbc~=0, stop')

    file.close()
    return coils(X,Y,Z,has_current,coil_number,n_nodes)



def read_currents(current_file):
    file = open(current_file, 'r')
    currents = [float(data) for data in file.read().split()]
    file.close()
    return currents



def read_grid(grid_file, L1i):
    f1=open(grid_file,'r')
    nr,np,nz=[int(data) for data in f1.readline().split()] # number of grid points in r, phi, z direction
    rmin, rmax=[float(data) for data in f1.readline().split()] # min and max values for r
    zmin, zmax=[float(data) for data in f1.readline().split()] # min and max values for z
    f1.close()
    phi_min=0
    phi_max=2*math.pi/L1i
    return make_grid(nr,np,nz, rmin, rmax, phi_min, phi_max, zmin, zmax)


def write_field_to_file(field_file, grid, B, L1i):
    file=open(field_file,'w')
    # Write the input parameters for the magnetic field calculation to the output file.
    file.write(f"{grid.n_R} {grid.n_phi} {grid.n_Z} {L1i}\n")
    file.write(f"{grid.R_min} {grid.R_max}\n")
    file.write(f"{grid.phi_min} {grid.phi_max}\n")
    file.write(f"{grid.Z_min} {grid.Z_max}\n")
    # Loop over the grid points and calculate the magnetic field components.
    for b in B:
        file.write(f"{b[0]} {b[1]} {b[2]}\n")
    file.close()

def get_field_on_grid(grid, coils, currents):
    B=[]
    for r in grid.R:
        for p in grid.phi:
            for z in grid.Z:
                x = [r,p,z] # coordinates for current grid point
                B_R, B_phi, B_Z=calc_biotsavart(x, coils, currents)
                B.append([B_R, B_phi, B_Z])
    return B

def make_field_file_from_coils(grid_file='biotsavart.inp', coil_file='co_asd.dd', current_file='cur_asd.dd', field_file='field.dat', L1i=1):
    coils = read_coils(coil_file)
    currents = read_currents(current_file)
    #
    grid = read_grid(grid_file, L1i)
    #
    B = get_field_on_grid(grid,coils,currents)
    #
    write_field_to_file(field_file, grid, B, L1i)

    
if __name__=="__main__":
    make_field_file_from_coils(grid_file='biotsavart.inp',coil_file='co_asd.dd',current_file='cur_asd.dd',field_file='field.dat',L1i=1)