#%%
import numpy as np
ier=0
PROP=False
facbc=1.0
#% facb=None

def gbcoil(RI,fI,ZI, coil_data):
    """Multiply values obtained by GBmodc with facbc"""
    global facbc
    global ier

    if ier!=1:
        f36 = open('fort.36', 'w')

    BX,Bf,BY,BRR,BRf,BRZ,BfR,Bff,BfZ,BZR,BZf,BZZ, nnodc = GBmodc(RI,fI,ZI, coil_data)

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

    if ier==0:
        print(BRI,'=br',BfI,'=bf',BZI,'=bz', file=f36)
        f36.close()
        ier=1

    return BRI,BfI,BZI,BRRI,BRfI,BRZI,BfRI,BffI,BfZI,BZRI,BZfI,BZZI


def GBmodc(RI,fI,ZI, coil_data):
    """Evaluate the biotsavart integral
    
    RI, fI, ZI - grid point cylindrical coordinates
    coil_data - coordinates of coil points
    """
    # w7x version

    #     GBX1

    # 19.05.2011
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
        coil0=[XO[0],YO[0],ZO[0]]
        R1_vector=np.subtract(grid_point,coil0)

        R1=np.linalg.norm(R1_vector)  #distance between grid point and first coil point | |r-r'[0]|
        R1X=XMXC/R1
        R1Y=YMYC/R1
        R1Z=ZMZC/R1
        e_R1=np.dot(R1_vector, 1/R1)
        for K in range(1,nnodc):   # loops through the remaining coil points
            AX=XO[K]-XO[K-1]  #difference between the current coil point and the previous one | l
            AY=YO[K]-YO[K-1]
            AZ=ZO[K]-ZO[K-1]
            XMXC=X-XO[K]        #difference between grid point and current coil point | r-r'[k]
            YMYC=Y-YO[K]
            ZMZC=Z-ZO[K]

            ZPRA=AX*XMXC+AY*YMYC+AZ*ZMZC    #scalar product of l and r-r'[k]
            R2=np.sqrt(XMXC*XMXC+YMYC*YMYC+ZMZC*ZMZC)   #distance between grid point and current coil point | |r-r'[k]|
            R2X=XMXC/R2
            R2Y=YMYC/R2
            R2Z=ZMZC/R2

            if cur[K-1]!=0.0:   # if not first point of a coil

                OBCP=1.0/(R2*(R1+R2)+ZPRA)
                FAZRDA=-(R1+R2)*OBCP/R1/R2
                FLR1=-R2*OBCP+1/(R1+R2)-1/R1
                FLR2=-(2.0*R2+R1)*OBCP+1/(R1+R2)-1/R2
                FLX=FLR1*R1X+FLR2*R2X-OBCP*AX
                FLY=FLR1*R1Y+FLR2*R2Y-OBCP*AY
                FLZ=FLR1*R1Z+FLR2*R2Z-OBCP*AZ
                BXB1=YMYC*AZ-ZMZC*AY    #r-r'[k] x l
                BYB1=ZMZC*AX-XMXC*AZ
                BZB1=XMXC*AY-YMYC*AX

                # 19.05.2011    FAZRDA=FAZRDA*cur[K-1]
                ncoi=nco[K]
                FAZRDA=FAZRDA*cur[K-1]*curco[ncoi-1]
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
    """Return the values in co_asd.dd and cur_asd.dd
    
    ncoil - lnumber of coils    
    XO, YO, ZO - coodinates of points on coil
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
    """
    #
    # Computes magnetic field module in units of the magnetic code  - bmod,
    # square root of determinant of the metric tensor           - sqrtg,
    # derivatives of the logarythm of the magnetic field module
    # over coordinates                                - bder,
    # covariant componets of the unit vector of the magnetic
    # field direction                                 - hcovar,
    # contravariant components of this vector                 - hctrvr,
    # contravariant component of the curl of this vector        - hcurl
    # Order of coordinates is the following: x[0]=R (big radius),
    # x[1]=phi (toroidal angle), x[2]=Z (altitude).
    #
    #  Input parameters:
    #        formal:  x            -    array of coordinates
    #  Output parameters:
    #        formal:  bmod
    #               sqrtg
    #               bder
    #               hcovar
    #               hctrvr
    #               hcurl
    #
    #  Called routines:  GBhs,GBRZd
    #
    rbig=max(x[0],1.0e-12)
    #
    ######## computation of gb in cylindrical co-ordinates ########
    ri=rbig
    fii=x[1]
    zi=x[2]

    br,bf,bz,brr,brf,brz,bfr,bff,bfz,bzr,bzf,bzz=gbcoil(ri,fii,zi, coil_data)

    ########## end of gb computation ##########
    bmod=np.sqrt(br**2+bf**2+bz**2)
    sqrtg=rbig
    hr=br/bmod
    hf=bf/bmod
    hz=bz/bmod
    #
    bder=np.empty(3)
    bder[0]=(brr*hr+bfr*hf+bzr*hz)/bmod
    bder[1]=(brf*hr+bff*hf+bzf*hz)/bmod
    bder[2]=(brz*hr+bfz*hf+bzz*hz)/bmod
    #
    hcovar=np.empty(3)
    hcovar[0]=hr
    hcovar[1]=hf*rbig
    hcovar[2]=hz
    #
    hctrvr=np.empty(3)
    hctrvr[0]=hr
    hctrvr[1]=hf/rbig
    hctrvr[2]=hz
    #
    hcurl=np.empty(3)
    hcurl[0]=((bzf-rbig*bfz)/bmod+
            hcovar[1]*bder[2]-hcovar[2]*bder[1])/sqrtg
    hcurl[1]=((brz-bzr)/bmod+
            hcovar[2]*bder[0]-hcovar[0]*bder[2])/sqrtg
    hcurl[2]=((bf+rbig*bfr-brf)/bmod+
            hcovar[0]*bder[1]-hcovar[1]*bder[0])/sqrtg
    #
    return bmod,sqrtg,bder,hcovar,hctrvr,hcurl
    #
