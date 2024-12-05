# Dominik Haberz 11916636
# Clara Rinner 01137166
#%%
# program biotsavart
import numpy as np
from grid import grid
import argparse

class coils:
    def __init__(self,X,Y,Z,has_current,coil_number,n_nodes):
        self.X=X
        self.Y=Y
        self.Z=Z
        self.has_current=has_current
        self.coil_number=coil_number
        self.n_nodes=n_nodes

def calc_biotsavart(grid_coordinates, coils, currents):
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

    coil_point=[coils.X[0],coils.Y[0],coils.Z[0]]
    R1_vector=np.subtract(grid_point,coil_point)  #difference of grid point and first coil point | r-r'[0]

    R1=np.linalg.norm(R1_vector)  #distance between grid point and first coil point | |r-r'[0]|

    for K in range(1,coils.n_nodes):   # loops through the remaining coil points

        coil_point_previous=coil_point
        coil_point=[coils.X[K],coils.Y[K],coils.Z[K]]

        L=np.subtract(coil_point, coil_point_previous)  #difference between the current coil point and the previous one | l

        R2_vector=np.subtract(grid_point,coil_point)#difference between grid point and current coil point | r-r'[k]

        scalar_product = np.dot(L, R2_vector)   #scalar product of l and r-r'[k]
        R2=np.linalg.norm(R2_vector)  #distance between grid point and current coil point | |r-r'[k]|


        if coils.has_current[K-1]!=0.0:   # if not first point of a coil

            factor1=1.0/(R2*(R1+R2)+scalar_product)
            factor2=-(R1+R2)*factor1/R1/R2*currents[coils.coil_number[K]-1] #curco[nco[K]-1] - current in coil
            cross_product=np.cross(R2_vector,L)    #r-r'[k] x l

            B=np.add(np.dot(cross_product, factor2), B)

        R1=R2

    B_R=B[0]*cosf+B[1]*sinf
    B_phi=B[1]*cosf-B[0]*sinf
    B_Z = B[2]

    return B_R, B_phi, B_Z

def calc_biotsavart_vectorized(grid_coordinates, coils, currents):
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


    coil_points=np.zeros((coils.n_nodes, 3))
    coil_points[:,0]=coils.X
    coil_points[:,1]=coils.Y
    coil_points[:,2]=coils.Z
    R_vectors=np.subtract(grid_point,coil_points)

    R = np.linalg.norm(R_vectors, axis=1)

    L_vectors = np.subtract(coil_points[1:], coil_points[:-1])

    scalar_products = np.einsum('ij,ij->i', L_vectors, R_vectors[1:])

    current_list = [currents[i-1] for i in coils.coil_number[1:]]

    factor1_list=1/(R[1:]*(R[:-1]+R[1:])+scalar_products)
    factor2_list=-(R[:-1]+R[1:])*factor1_list/R[:-1]/R[1:]*coils.has_current[:-1]*current_list
    cross_products=np.cross(R_vectors[1:], L_vectors)

    B = np.sum(cross_products*factor2_list[:,np.newaxis], axis=0)

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
    nR,nphi,nZ=[int(data) for data in f1.readline().split()] # number of grid points in r, phi, z direction
    R_min, R_max=[float(data) for data in f1.readline().split()] # min and max values for r
    phi_min = 0
    phi_max = 2*np.pi/L1i
    Z_min, Z_max=[float(data) for data in f1.readline().split()] # min and max values for z
    f1.close()

    return grid(nR, nphi, nZ, R_min, R_max, phi_min, phi_max, Z_min, Z_max)


def write_field_to_file(field_file, grid, B, L1i):
    file=open(field_file,'w')
    # Write the input parameters for the magnetic field calculation to the output file.
    file.write(f"{grid.nR} {grid.nphi} {grid.nZ} {L1i}\n")
    file.write(f"{grid.R_min} {grid.R_max}\n")
    file.write(f"{grid.phi_min} {grid.phi_max}\n")
    file.write(f"{grid.Z_min} {grid.Z_max}\n")
    # Loop over the grid points and calculate the magnetic field components.
    for b in B:
        file.write(f"{b[0]} {b[1]} {b[2]}\n")
    file.close()

def get_field_on_grid(grid, coils, currents, integrator):
    B=[]
    for r in grid.R:
        for p in grid.phi:
            for z in grid.Z:
                x = [r,p,z] # coordinates for current grid point
                B_R, B_phi, B_Z=integrator(x, coils, currents)
                B.append([B_R, B_phi, B_Z])
    return B

def make_field_file_from_coils(grid_file='biotsavart.inp', coil_file='co_asd.dd', current_file='cur_asd.dd', field_file='field.dat', L1i=1):
    coils = read_coils(coil_file)
    currents = read_currents(current_file)
    #
    grid = read_grid(grid_file, L1i)
    #
    B = get_field_on_grid(grid,coils,currents,calc_biotsavart)
    #
    write_field_to_file(field_file, grid, B, L1i)

    
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--grid_file', default='biotsavart.inp')
    parser.add_argument('--coil_file', default='co_asd.dd')
    parser.add_argument('--current_file', default='cur_asd.dd')
    parser.add_argument('--field_file', default='field.dat')
    parser.add_argument('--L1i', type=int,default=1)
    args=parser.parse_args()
    make_field_file_from_coils(args.grid_file, args.coil_file, args.current_file, args.field_file, args.L1i)
