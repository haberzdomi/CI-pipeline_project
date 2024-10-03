#%%
import numpy as np
ier=0
PROP=False
facbc=1.0
#% facb=None

def gbcoil(RI,fI,ZI):

    global facbc, nnodc
    global facb, ier

    if ier!=1:
        facb=facbc
        f36 = open('fort.36', 'w')

    X=RI
    Y=ZI

    BX,Bf,BY,BRR,BRf,BRZ,BfR,Bff,BfZ,BZR,BZf,BZZ = GBmodc(X,fI,Y)

    if ier==0:
        print(facb,'=facb',nnodc,'=nnodc', file=f36)
        print(X,'=r_st',fI,'=fi_st',Y,'=z_st', file=f36)

    BRI=BX*facb
    BfI=Bf*facb
    BZI=BY*facb
    BRRI=BRR*facb
    BRfI=BRf*facb
    BRZI=BRZ*facb
    BfRI=BfR*facb
    BffI=Bff*facb
    BfZI=BfZ*facb
    BZRI=BZR*facb
    BZfI=BZf*facb
    BZZI=BZZ*facb

    if ier==0:
        print(BRI,'=br',BfI,'=bf',BZI,'=bz', file=f36)
        f36.close()
        ier=1

    return BRI,BfI,BZI,BRRI,BRfI,BRZI,BfRI,BffI,BfZI,BZRI,BZfI,BZZI


def GBmodc(RI,fI,ZI):
    # w7x version

    # 25.07.2012    npar,ncoil=10850,27
    npar,ncoil=10850,16
    # 25.07.2012    file2='cur_it.dd'
    file2='cur_asd.dd'
    # 25.07.2012    file3='co_itm.dd'
    file3='co_asd.dd'

    global curco
    #     GBX1

    global nnodc
    # 19.05.2011
    global PROP, bfrt, NPR, K2, XO, YO, ZO, cur, nco
    if not PROP:
        PROP=True

        bfrt=0.0

        nparx=npar
        XO,YO,ZO,cur,nco=datw7xm(nparx)

        NPR=1
        K2=nnodc


        print(' gbrfz from GBXOT   (r*8, GBX1) bfrt=%#17.10E' % bfrt)
        print('  w7x_g version')
        print('curco')
        print(curco)

    fd=fI
    rd=RI
    cosf=np.cos(fd)
    sinf=np.sin(fd)
    Y=rd*sinf
    X=rd*cosf
    Z=ZI
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
    K0=0
    K1=1
    for N in range(NPR):
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
        XMXC=X-XO[K0]
        YMYC=Y-YO[K0]
        ZMZC=Z-ZO[K0]

        R1=np.sqrt(XMXC*XMXC+YMYC*YMYC+ZMZC*ZMZC)
        OR1=1.0/R1
        R1X=XMXC*OR1
        R1Y=YMYC*OR1
        R1Z=ZMZC*OR1
        for K in range(K1,K2):
            AX=XO[K]-XO[K-1]
            XMXC=X-XO[K]
            AY=YO[K]-YO[K-1]
            YMYC=Y-YO[K]

            AZ=ZO[K]-ZO[K-1]
            ZMZC=Z-ZO[K]
            ZPRA=AX*XMXC+AY*YMYC+AZ*ZMZC
            R2=np.sqrt(XMXC*XMXC+YMYC*YMYC+ZMZC*ZMZC)
            OR2=1.0/R2
            R2X=XMXC*OR2
            R2Y=YMYC*OR2
            R2Z=ZMZC*OR2

            R1PR2=R1+R2
            OR1PR2=1.0/R1PR2

            if cur[K-1]!=0.0:

                OBCP=1.0/(R2*R1PR2+ZPRA)
                FAZRDA=-R1PR2*OBCP*OR1*OR2
                FLZA=-OBCP
                FLR1=-R2*OBCP+OR1PR2-OR1
                FLR2=-(2.0*R2+R1)*OBCP+OR1PR2-OR2
                FLX=FLR1*R1X+FLR2*R2X+FLZA*AX
                FLY=FLR1*R1Y+FLR2*R2Y+FLZA*AY
                FLZ=FLR1*R1Z+FLR2*R2Z+FLZA*AZ
                BXB1=YMYC*AZ-ZMZC*AY
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
            OR1=OR2

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
    BfB=bfrt/rd
    Bf=BY*cosf-BX*sinf
    BfI=Bf+BfB
    # 25.12.2008    bxy2sc=BXY*sicof*2.0
    bxybyx=(BXY+BYX)*sicof
    byybxx=(BYY-BXX)*sicof
    BRRI=BXX*cosf2+BYY*sinf2+bxybyx
    BfR=BYX*cosf2-BXY*sinf2+byybxx
    BfRI=BfR-BfB/rd
    BZRI=BZX*cosf+BZY*sinf
    BRfI=(BXY*cosf2-BYX*sinf2+byybxx)*rd+Bf
    BffI=(BYY*cosf2+BXX*sinf2-bxybyx)*rd-BR
    BZfI=(BZY*cosf-BZX*sinf)*rd
    BRZI=BXZ*cosf+BYZ*sinf
    BfZI=BYZ*cosf-BXZ*sinf
    # 25.12.2008 end
    BZI=BZ
    BZZI=BZZ

    return BRI,BfI,BZI,BRRI,BRfI,BRZI,BfRI,BffI,BfZI,BZRI,BZfI,BZZI


#     def datw7xm(nparx):
def datw7xm(neli):

    XO = np.empty(neli)
    YO = np.empty(neli)
    ZO = np.empty(neli)
    cur = np.empty(neli)
    # 25.07.2012    npar,ncoil=10850,27
    npar,ncoil=10850,16
	  # 25.07.2012    file2='cur_it.dd'
    file2='cur_asd.dd'
    # 25.07.2012    file3='co_itm.dd'
    file3='co_asd.dd'

    global curco
    curco = np.empty(ncoil)
    nco = np.empty(neli,dtype=int)
    global nnodc
    # 19.05.2011

    f1 = open(file3, 'r')
    # 19.05.2011
    f2 = open(file2, 'r')
    curco[:] = [float(data) for data in f2.read().split()]
    # 19.05.2011 end
    nnod = int(f1.readline())
    nnodc=nnod

    if nnod>neli:
        print(nnod,'=nnod',neli,'=neli')
        raise Exception('nnod>neli, stop')

    for knod in range(nnod):
        XO[knod],YO[knod],ZO[knod],cur[knod],nco[knod] = [convert(data)
            for convert, data in zip([float,float,float,float,int], f1.readline().split())]
    cbc=cur[nnod-1]


    if cbc!=0.0:
        raise Exception(str(cbc)+'=cbc~=0, stop')

    print(nnod,'=nnod,  datw7x terminated ')
    f1.close()
    # 25.07.2012
    f2.close()
    # 25.07.2012 end
    return XO,YO,ZO,cur,nco

#
#
#
def magfie(x):
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

    br,bf,bz,brr,brf,brz,bfr,bff,bfz,bzr,bzf,bzz=gbcoil(ri,fii,zi)

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
