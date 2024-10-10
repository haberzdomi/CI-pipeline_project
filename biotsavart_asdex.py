# Dominik Haberz 11916636
# Clara Rinner 01137166
#%%
# program biotsavart
#
from vvn_asdex import magfie, datw7xm
#%%
import math
ncoil=16
coil_data = datw7xm(ncoil)
x = [0.0, 0.0, 0.0]
#% global facbc
#
#% facbc=1.0
#
f1=open('biotsavart.inp','r')
nr,np,nz=[int(data) for data in f1.readline().split()]
rmin, rmax=[float(data) for data in f1.readline().split()]
zmin, zmax=[float(data) for data in f1.readline().split()]
f1.close()
#
L1i=1
#
pmin=0.0
pmax=2*math.pi/L1i
#
hrad = (rmax - rmin)/(nr-1)
hphi = (pmax - pmin)/(np-1)
hzet = (zmax - zmin)/(nz-1)
#
#%%
f1=open('field.dat','w')
print(nr,np,nz,L1i,file=f1)
print(rmin,rmax,file=f1)
print(pmin,pmax,file=f1)
print(zmin,zmax,file=f1)
for i in range(nr):
    print(i+1,'/',nr)
    for j in range(np):
        for k in range(nz):
            x = [rmin+hrad*i,pmin+hphi*j,zmin+hzet*k]
            bmod,sqrtg,bder,hcovar,hctrvr,hcurl=magfie(x, coil_data)
            print(hcovar[0]*bmod,hcovar[1]*bmod/x[0],hcovar[2]*bmod,file=f1)
f1.close()
#