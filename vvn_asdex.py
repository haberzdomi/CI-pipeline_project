#%%
import numpy as np
# If ier is set to 1 the file fort.36 will not be created or written to and 
# facb has to be given for the magnetic field scaling factor.
ier=0
# If PROP is set to False the function datw7xm will be called to read the coil 
# data from the files cur_asd.dd and co_asd.dd. In the current version of this 
# module PROP=True will cause an error in GBmodc. 
PROP=False
# Magnetic field scaling factor
facbc=1.0 
#% facb=None

def gbcoil(RI,fI,ZI, coil_data):
    """
    Get the magnetic field components and their derivatives from GBmodc and 
    scale them with the factor facb. Also create the file fort.36 containing 
    the magnetic field scaling factor (facb), total number of nodes for all 
    coils (nnodc), radius (r_st), the toroidal angle (fi_st), the altitude (z_st) 
    and magnetic field in r, phi and z-direction (br, b_f, bz).
    
    Args:
        RI (float): big radius
        fI (float): toroidal angle
        ZI (float): altitude
        coil_data - coordinates of the coil points

    Returns:
        BRI (float): radial component fo the magnetic field
        BfI (float): toroidal component of the magnetic field
        BZI (float): axial component of the magnetic field
        BRRI (float): derivative of BRI in radial direction
        BRfI (float): derivative of BRI in toroidal direction
        BRZI (float): derivative of BRI in axial direction
        BfRI (float): derivative of BfI in radial direction
        BffI (float): derivative of BfI in toroidal direction
        BfZI (float): derivative of BfI in axial direction
        BZRI (float): derivative of BZI in radial direction 
        BZfI (float): derivative of BZI in toroidal direction 
        BZZI (float): derivative of BZI in axial direction 
    """
    global facbc
    global ier

    # Skip if ier is 1, otherwise set facb to facbc and create fort.36.
    if ier!=1:
        f36 = open('fort.36', 'w')

    BX,Bf,BY,BRR,BRf,BRZ,BfR,Bff,BfZ,BZR,BZf,BZZ, nnodc = GBmodc(RI,fI,ZI, coil_data)

    # Skip if ier is 1, otherwise write some input parameters for the magnetic field calculation to fort.36.
    if ier==0:
        print(facbc,'=facb',nnodc,'=nnodc', file=f36)
        print(RI,'=r_st',fI,'=fi_st',ZI,'=z_st', file=f36)

    BRI=BX*facbc
    BfI=Bf*facbc
    BZI=BY*facbc
    BRRI=BRR*facbc
    BRfI=BRf*facbc
    BRZI=BRZ*facbc
    BfRI=BfR*facbc
    BffI=Bff*facbc
    BfZI=BfZ*facbc
    BZRI=BZR*facbc
    BZfI=BZf*facbc
    BZZI=BZZ*facbc

    # Skip if ier is 1, otherwise write the magnetic field components to the file fort.36 and close it.
    if ier==0:
        print(BRI,'=br',BfI,'=bf',BZI,'=bz', file=f36)
        f36.close()
        ier=1

    return BRI,BfI,BZI,BRRI,BRfI,BRZI,BfRI,BffI,BfZI,BZRI,BZfI,BZZI


def GBmodc(RI,fI,ZI, coil_data):
    """
    Calculate the magnetic field components and their derivatives by 
    evaluating the Biot-Savart integral. The geometry and current values 
    of the coil are read from the files cur_asd.dd and co_asd.dd by 
    calling datw7xm.

    Args:
        RI (float): big radius
        fI (float): toroidal angle
        ZI (float): altitude
        coil_data - coordinates of the coil points

    Returns:
        BRI (float): radial component fo the magnetic field
        BfI (float): toroidal component of the magnetic field
        BZI (float): axial component of the magnetic field
        BRRI (float): derivative of BRI in radial direction
        BRfI (float): derivative of BRI in toroidal direction
        BRZI (float): derivative of BRI in axial direction
        BfRI (float): derivative of BfI in radial direction
        BffI (float): derivative of BfI in toroidal direction
        BfZI (float): derivative of BfI in axial direction
        BZRI (float): derivative of BZI in radial direction 
        BZfI (float): derivative of BZI in toroidal direction 
        BZZI (float): derivative of BZI in axial direction
    """
    global PROP, bfrt, NPR

    [XO,YO,ZO,cur,nco,nnodc,curco]=coil_data

    if not PROP:
        PROP=True

        bfrt=0.0

        NPR=1

        print(' gbrfz from GBXOT   (r*8, GBX1) bfrt=%#17.10E' % bfrt)
        print('  w7x_g version')
        print('curco')
        print(curco)
    

    cosf=np.cos(fI)
    sinf=np.sin(fI)
    Y=RI*sinf  #cartesian coordinates of grid point
    X=RI*cosf
    Z=ZI
    grid_point=[X,Y,Z]
    BX=0.0
    BY=0.0
    BZ=0.0
    BXX=0.0
    BXY=0.0

    BXZ=0.0
    BYY=0.0
    BYZ=0.0
    BZZ=0.0
    # 25.12.2008
    BYX=0.0
    BZX=0.0
    BZY=0.0
    # 25.12.2008 end
    for N in range(NPR):  #NPR=1
        BXB=0.0
        BYB=0.0

        BZB=0.0
        BXXB=0.0
        BXYB=0.0
        BXZB=0.0
        BYYB=0.0
        BYZB=0.0
        BZZB=0.0
        # 25.12.2008
        BYXB=0.0
        BZXB=0.0
        BZYB=0.0
        # 25.12.2008 end
        XMXC=X-XO[0]  #difference of grid point and first coil point | r-r'[0]
        YMYC=Y-YO[0]
        ZMZC=Z-ZO[0]
        coil_point_current=[XO[0],YO[0],ZO[0]]
        R1_vector=np.subtract(grid_point,coil_point_current)

        R1=np.linalg.norm(R1_vector)  #distance between grid point and first coil point | |r-r'[0]|
        R1X=XMXC/R1
        R1Y=YMYC/R1
        R1Z=ZMZC/R1
        e_R1=np.dot(R1_vector, 1/R1)
        for K in range(1,nnodc):   # loops through the remaining coil points
            AX=XO[K]-XO[K-1]  #difference between the current coil point and the previous one | l
            AY=YO[K]-YO[K-1]
            AZ=ZO[K]-ZO[K-1]
            coil_point_previous=coil_point_current
            coil_point_current=[XO[K],YO[K],ZO[K]]

            A=np.subtract(coil_point_current, coil_point_previous)
            XMXC=X-XO[K]        #difference between grid point and current coil point | r-r'[k]
            YMYC=Y-YO[K]
            ZMZC=Z-ZO[K]
            R2_vector=np.subtract(grid_point,coil_point_current)

            ZPRA=AX*XMXC+AY*YMYC+AZ*ZMZC    #scalar product of l and r-r'[k]
            R2=np.linalg.norm(R2_vector)  #distance between grid point and current coil point | |r-r'[k]|
            R2X=XMXC/R2
            R2Y=YMYC/R2
            R2Z=ZMZC/R2
            e_R2=np.dot(R2_vector, 1/R2)

            if cur[K-1]!=0.0:   # if not first point of a coil

                OBCP=1.0/(R2*(R1+R2)+ZPRA)
                FAZRDA=-(R1+R2)*OBCP/R1/R2
                FLR1=-R2*OBCP+1/(R1+R2)-1/R1
                FLR2=-(2.0*R2+R1)*OBCP+1/(R1+R2)-1/R2
                FLX=FLR1*R1X+FLR2*R2X-OBCP*AX
                FLY=FLR1*R1Y+FLR2*R2Y-OBCP*AY
                FLZ=FLR1*R1Z+FLR2*R2Z-OBCP*AZ
                FL_=np.add(np.dot(FLR1, e_R1), np.dot(FLR2, e_R2), np.dot(-OBCP,A))
                BXB1=YMYC*AZ-ZMZC*AY    #r-r'[k] x l
                BYB1=ZMZC*AX-XMXC*AZ
                BZB1=XMXC*AY-YMYC*AX
                B_B1=np.cross(R2_vector,A)

                # 19.05.2011    FAZRDA=FAZRDA*cur[K-1]
                ncoi=nco[K]
                FAZRDA=FAZRDA*cur[K-1]*curco[ncoi-1] #cur[K-1]=1    curco[ncoi-1] - current in coil
                # 19.05.2011 end

                BXB=BXB1*FAZRDA+BXB
                BYB=BYB1*FAZRDA+BYB
                BZB=BZB1*FAZRDA+BZB
                BXXB=FLX*BXB1*FAZRDA+BXXB
                BXYB=(FLY*BXB1+AZ)*FAZRDA+BXYB
                BXZB=(FLZ*BXB1-AY)*FAZRDA+BXZB
                BYYB=FLY*BYB1*FAZRDA+BYYB
                BYZB=(FLZ*BYB1+AX)*FAZRDA+BYZB
                BZZB=FLZ*BZB1*FAZRDA+BZZB
                # 25.12.2008
                BYXB=(FLX*BYB1-AZ)*FAZRDA+BYXB
                BZXB=(FLX*BZB1+AY)*FAZRDA+BZXB
                BZYB=(FLY*BZB1-AX)*FAZRDA+BZYB
                # 25.12.2008 end

            R1=R2
            R1X=R2X
            R1Y=R2Y
            R1Z=R2Z

        BX=BXB
        BY=BYB
        BZ=BZB
        BXX=BXXB
        BXY=BXYB
        BXZ=BXZB
        BYY=BYYB
        BYZ=BYZB
        BZZ=BZZB
        # 25.12.2008
        BYX=BYXB
        BZX=BZXB
        BZY=BZYB
        # 25.12.2008 end
    cosf2=cosf*cosf
    sinf2=sinf*sinf
    sicof=sinf*cosf
    BR=BX*cosf+BY*sinf
    BRI=BR
    BfB=bfrt/RI
    Bf=BY*cosf-BX*sinf
    BfI=Bf+BfB
    # 25.12.2008    bxy2sc=BXY*sicof*2.0
    bxybyx=(BXY+BYX)*sicof
    byybxx=(BYY-BXX)*sicof
    BRRI=BXX*cosf2+BYY*sinf2+bxybyx
    BfR=BYX*cosf2-BXY*sinf2+byybxx
    BfRI=BfR-BfB/RI
    BZRI=BZX*cosf+BZY*sinf
    BRfI=(BXY*cosf2-BYX*sinf2+byybxx)*RI+Bf
    BffI=(BYY*cosf2+BXX*sinf2-bxybyx)*RI-BR
    BZfI=(BZY*cosf-BZX*sinf)*RI
    BRZI=BXZ*cosf+BYZ*sinf
    BfZI=BYZ*cosf-BXZ*sinf
    # 25.12.2008 end
    BZI=BZ
    BZZI=BZZ

    return BRI,BfI,BZI,BRRI,BRfI,BRZI,BfRI,BffI,BfZI,BZRI,BZfI,BZZI,nnodc


#     def datw7xm(nparx):
def load_coil_data(ncoil):
    """
    Return the data in co_asd.dd and cur_asd.dd
    
    Args: 
        ncoil - number of coils  
    Returns:
        XO, YO, ZO - coodinate lists of coil points
        cur - 0 if last point in a coil, otherwise 1
        nco - coil number
        nnodc - number of coil points
        curco - values in cur_asd.dd
    """

    # 19.05.2011

    f1 = open('co_asd.dd', 'r')
    # 19.05.2011
    f2 = open('cur_asd.dd', 'r')
    # 19.05.2011 end
    nnodc = int(f1.readline())

    XO = np.empty(nnodc)
    YO = np.empty(nnodc)
    ZO = np.empty(nnodc)
    cur = np.empty(nnodc)

    curco = np.empty(ncoil)
    nco = np.empty(nnodc,dtype=int)

    curco[:] = [float(data) for data in f2.read().split()] #Current?


    for knod in range(nnodc):
        XO[knod],YO[knod],ZO[knod],cur[knod],nco[knod] = [convert(data)
            for convert, data in zip([float,float,float,float,int], f1.readline().split())]
    cbc=cur[nnodc-1]


    # Raise error if the current factor for the last node of the last coil is not 0.
    if cbc!=0.0:
        raise Exception(str(cbc)+'=cbc~=0, stop')

    print(nnodc,'=nnod,  datw7x terminated ')
    f1.close()
    # 25.07.2012
    f2.close()
    # 25.07.2012 end
    return [XO,YO,ZO,cur,nco,nnodc,curco]

#
#
#
def magfie(x, coil_data):
    """
    Get the magnetic field components and their derivatives from gbcoil and use them to determine the below 
    properties (returns).  

    Args:
        x (List[float], size=3): cylindrical coordinates of grid point where the magnetic field is to be determined
                                 x[0]=R (big radius), x[1]=phi (toroidal angle), x[2]=Z (altitude)
    
    Returns:
        bmod (float): magnetic field module in units of the magnetic code
        sqrtg (float): square root of determinant of the metric tensor
        bder (array[float], shape=(3,)): derivatives of the logarithm of the magnetic field module over coordinates
        hcovar (array[float], shape=(3,)): covariant components of the unit vector of the magnetic field direction
        hctrvr (array[float], shape=(3,)): contravariant components the unit vector of the magnetic field direction
        hcurl (array[float], shape=(3,)): contravariant component of the curl the unit vector of the magnetic field direction
    """
    
    rbig=max(x[0],1.0e-12)  # Prevent division by zero with minimum big radius R
    
    ######## computation of gb in cylindrical co-ordinates ########
    ri=rbig
    fii=x[1]
    zi=x[2]

    br,bf,bz,brr,brf,brz,bfr,bff,bfz,bzr,bzf,bzz=gbcoil(ri,fii,zi, coil_data)

    ########## end of gb computation ##########
    
    # For all following variable naming: 
    # r - radial component, f - toroidal component, z - axial component
    
    bmod=np.sqrt(br**2+bf**2+bz**2) # magnetic field module
    sqrtg=rbig # square root of determinant of the metric tensor
    
    # unit vectors of the magnetic field directions
    hr=br/bmod
    hf=bf/bmod
    hz=bz/bmod
    
    # derivatives of the logarithm of the magnetic field module over coordinates
    bder=np.empty(3)
    bder[0]=(brr*hr+bfr*hf+bzr*hz)/bmod
    bder[1]=(brf*hr+bff*hf+bzf*hz)/bmod
    bder[2]=(brz*hr+bfz*hf+bzz*hz)/bmod
    
    # covariant componets of the unit vector of the magnetic field direction
    hcovar=np.empty(3)
    hcovar[0]=hr
    hcovar[1]=hf*rbig
    hcovar[2]=hz
    
    # contravariant components of the unit vector of the magnetic field direction
    hctrvr=np.empty(3)
    hctrvr[0]=hr
    hctrvr[1]=hf/rbig
    hctrvr[2]=hz
    
    # contravariant component of the curl of the unit vector of the magnetic field direction
    hcurl=np.empty(3)
    hcurl[0]=((bzf-rbig*bfz)/bmod+
            hcovar[1]*bder[2]-hcovar[2]*bder[1])/sqrtg
    hcurl[1]=((brz-bzr)/bmod+
            hcovar[2]*bder[0]-hcovar[0]*bder[2])/sqrtg
    hcurl[2]=((bf+rbig*bfr-brf)/bmod+
            hcovar[0]*bder[1]-hcovar[1]*bder[0])/sqrtg
    
    return bmod,sqrtg,bder,hcovar,hctrvr,hcurl

