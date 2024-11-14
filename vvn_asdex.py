#%%
import numpy as np


def calculate_magnetic_field(grid_coordinates, coil_data):
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

    [XO,YO,ZO,cur,nco,nnodc,curco]=coil_data

    RI=grid_coordinates[0]
    fI=grid_coordinates[1]
    ZI=grid_coordinates[2]
    cosf=np.cos(fI)
    sinf=np.sin(fI)
    Y=RI*sinf  #cartesian coordinates of grid point
    X=RI*cosf
    Z=ZI
    grid_point=[X,Y,Z]

    B=[0,0,0]

    coil_point_current=[XO[0],YO[0],ZO[0]]
    R1_vector=np.subtract(grid_point,coil_point_current)  #difference of grid point and first coil point | r-r'[0]

    R1=np.linalg.norm(R1_vector)  #distance between grid point and first coil point | |r-r'[0]|

    for K in range(1,nnodc):   # loops through the remaining coil points

        coil_point_previous=coil_point_current
        coil_point_current=[XO[K],YO[K],ZO[K]]

        A=np.subtract(coil_point_current, coil_point_previous)  #difference between the current coil point and the previous one | l

        R2_vector=np.subtract(grid_point,coil_point_current)#difference between grid point and current coil point | r-r'[k]

        scalar_product = np.dot(A, R2_vector)   #scalar product of l and r-r'[k]
        R2=np.linalg.norm(R2_vector)  #distance between grid point and current coil point | |r-r'[k]|


        if cur[K-1]!=0.0:   # if not first point of a coil

            ncoi=nco[K]
            OBCP=1.0/(R2*(R1+R2)+scalar_product)
            FAZRDA=-(R1+R2)*OBCP/R1/R2*curco[ncoi-1] #curco[ncoi-1] - current in coil
            B_B1=np.cross(R2_vector,A)    #r-r'[k] x l

            B=np.add(np.dot(B_B1, FAZRDA), B)

        R1=R2

    BR=B[0]*cosf+B[1]*sinf
    Bf=B[1]*cosf-B[0]*sinf
    B_cylinder = [BR, Bf, B[2]]

    return B_cylinder


def load_coil_data():
    """
    Return the data in co_asd.dd and cur_asd.dd
    
    Args: 
        None
    Returns:
        XO, YO, ZO - coodinate lists of coil points
        cur - 0 if last point in a coil, otherwise 1
        nco - coil number
        nnodc - number of coil points
        curco - values in cur_asd.dd
    """

    f1 = open('co_asd.dd', 'r')
    f2 = open('cur_asd.dd', 'r')
    nnodc = int(f1.readline())

    XO = np.empty(nnodc)
    YO = np.empty(nnodc)
    ZO = np.empty(nnodc)
    cur = np.empty(nnodc)

    nco = np.empty(nnodc,dtype=int)

    curco = [float(data) for data in f2.read().split()] #Current


    for knod in range(nnodc):
        XO[knod],YO[knod],ZO[knod],cur[knod],nco[knod] = [convert(data)
            for convert, data in zip([float,float,float,float,int], f1.readline().split())]
    cbc=cur[nnodc-1]


    # Raise error if the current factor for the last node of the last coil is not 0.
    if cbc!=0.0:
        raise Exception(str(cbc)+'=cbc~=0, stop')

    print(nnodc,'=nnod,  datw7x terminated ')
    f1.close()
    f2.close()
    return [XO,YO,ZO,cur,nco,nnodc,curco]

