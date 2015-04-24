from math import *
import numpy as np
import matplotlib.pyplot as plt
import random

N=64
Lbox = 64.

q1d = np.linspace(0.0, (N-1.)/N, N)*Lbox

############## 2D check #############
'''
xi, yi = np.meshgrid(q1d, q1d)

uxi = np.zeros_like(xi)
uyi = np.zeros_like(yi)

for i in range(N):
    for j in range(N):
        if i>32: 
            uxi[i,j] = 1.
        if i<32:
            uxi[i,j] = -1.
'''
############## 3D check #############
xi, yi, zi = np.meshgrid(q1d, q1d, q1d)

uxi = np.zeros_like(xi)
uyi = np.zeros_like(yi)
uzi = np.zeros_like(zi)

for i in range(N):
    for j in range(N):
        for k in range(N):
            if i>32: 
                uxi[i,j,k] = 1
            if i<32:
                uxi[i,j,k] = -1


############ DERIVATIVE AND CURL #############
def der(func,dir,x,y,z):
    
    h = 1.0

    ix = np.array(np.floor(x),dtype=int)
    iy = np.array(np.floor(y),dtype=int)
    iz = np.array(np.floor(z),dtype=int)    

    ix = (ix)%N; iy = (iy)%N; iz = (iz)%N;
    ix1 = (ix+1)%N; iy1 = (iy+1)%N; iz1 = (iz+1)%N;
    ix2 = (ix+2)%N; iy2 = (iy+2)%N; iz2 = (iz+2)%N;
    
    if dir == 'x':
        f = (8*(func[ix1,iy,iz] - func[ix-1,iy,iz]) + func[ix-2,iy,iz] - func[ix2,iy,iz])/(12.0*h)
    if dir == 'y':
        f = (8*(func[ix,iy1,iz] - func[ix,iy-1,iz]) + func[ix,iy-2,iz] - func[ix,iy2,iz])/(12.0*h)
    if dir == 'z':
        f = (8*(func[ix,iy,iz1] - func[ix,iy,iz-1]) + func[ix,iy,iz-2] - func[ix,iy,iz2])/(12.0*h)
    return f

def curl(x,y,z,fx,fy,fz):
    wx = der(fz,'y',x,y,z) - der(fy,'z',x,y,z)
    wy = der(fx,'z',x,y,z) - der(fz,'x',x,y,z)
    wz = der(fy,'x',x,y,z) - der(fx,'y',x,y,z)
    return wx,wy,wz

######### CIC interpolation ########

def CIC(x,y,z,vx1,vy1,vz1):
    
        vx = np.zeros((N,N,N))
        vy = np.zeros((N,N,N))
        vz = np.zeros((N,N,N))
        
        #vx = vx1.copy()
        #vy = vy1.copy()
        #vz = vz1.copy()

        
        ix = np.array(np.floor(x),dtype=int)
        iy = np.array(np.floor(y),dtype=int)
        iz = np.array(np.floor(z),dtype=int)    
        
        dx = x-ix
        dy = y-iy
        dz = z-iz

        tx = 1. - dx
        ty = 1. - dy
        tz = 1. - dz

        N2 = N
        
        
        for i in range(N**3):              
            
        #vx
            vx[ix[i]%N2,iy[i]%N2,iz[i]%N2] += tx[i]*ty[i]*tz[i]*vx1[i]
            vx[(ix[i]+1)%N2,iy[i]%N2,iz[i]%N2] += dx[i]*ty[i]*tz[i]*vx1[i]
            vx[ix[i]%N2,(iy[i]+1)%N2,iz[i]%N2] +=  tx[i]*dy[i]*tz[i]*vx1[i]
            vx[(ix[i]+1)%N2,(iy[i]+1)%N2,iz[i]%N2] +=  dx[i]*dy[i]*tz[i]*vx1[i]
            vx[ix[i]%N2,iy[i]%N2,(iz[i]+1)%N2] +=  tx[i]*ty[i]*dz[i]*vx1[i]
            vx[(ix[i]+1)%N2,iy[i]%N2,(iz[i]+1)%N2] +=  dx[i]*ty[i]*dz[i]*vx1[i]
            vx[ix[i]%N2,(iy[i]+1)%N2,(iz[i]+1)%N2] +=  tx[i]*dy[i]*dz[i]*vx1[i]
            vx[(ix[i]+1)%N2,(iy[i]+1)%N2,(iz[i]+1)%N2] +=  dx[i]*dy[i]*dz[i]*vx1[i]     
        
        #vy
            vy[ix[i]%N2,iy[i]%N2,iz[i]%N2] += tx[i]*ty[i]*tz[i]*vy1[i]
            vy[(ix[i]+1)%N2,iy[i]%N2,iz[i]%N2] += dx[i]*ty[i]*tz[i]*vy1[i]
            vy[ix[i]%N2,(iy[i]+1)%N2,iz[i]%N2] +=  tx[i]*dy[i]*tz[i]*vy1[i]
            vy[(ix[i]+1)%N2,(iy[i]+1)%N2,iz[i]%N2] +=  dx[i]*dy[i]*tz[i]*vy1[i]
            vy[ix[i]%N2,iy[i]%N2,(iz[i]+1)%N2] +=  tx[i]*ty[i]*dz[i]*vy1[i]
            vy[(ix[i]+1)%N2,iy[i]%N2,(iz[i]+1)%N2] +=  dx[i]*ty[i]*dz[i]*vy1[i]
            vy[ix[i]%N2,(iy[i]+1)%N2,(iz[i]+1)%N2] +=  tx[i]*dy[i]*dz[i]*vy1[i]
            vy[(ix[i]+1)%N2,(iy[i]+1)%N2,(iz[i]+1)%N2] +=  dx[i]*dy[i]*dz[i]*vy1[i]     
        
        #vz
            vz[ix[i]%N2,iy[i]%N2,iz[i]%N2] += tx[i]*ty[i]*tz[i]*vz1[i]
            vz[(ix[i]+1)%N2,iy[i]%N2,iz[i]%N2] += dx[i]*ty[i]*tz[i]*vz1[i]
            vz[ix[i]%N2,(iy[i]+1)%N2,iz[i]%N2] +=  tx[i]*dy[i]*tz[i]*vz1[i]
            vz[(ix[i]+1)%N2,(iy[i]+1)%N2,iz[i]%N2] +=  dx[i]*dy[i]*tz[i]*vz1[i]
            vz[ix[i]%N2,iy[i]%N2,(iz[i]+1)%N2] +=  tx[i]*ty[i]*dz[i]*vz1[i]
            vz[(ix[i]+1)%N2,iy[i]%N2,(iz[i]+1)%N2] +=  dx[i]*ty[i]*dz[i]*vz1[i]
            vz[ix[i]%N2,(iy[i]+1)%N2,(iz[i]+1)%N2] +=  tx[i]*dy[i]*dz[i]*vz1[i]
            vz[(ix[i]+1)%N2,(iy[i]+1)%N2,(iz[i]+1)%N2] +=  dx[i]*dy[i]*dz[i]*vz1[i]     
        
        return vx,vy,vz


########## VORTICITY ###########
#vx = px2 / a
#vy = py2 / a
#vz = pz2 / a

def computeVorticity(xi,yi,zi,uxi,uyi,uzi,n):
    xflat = np.ravel(xi)
    yflat = np.ravel(yi)
    zflat = np.ravel(zi)

    vxflat = np.ravel(uxi)
    vyflat = np.ravel(uyi)
    vzflat = np.ravel(uzi)

    Vx,Vy,Vz = CIC(xflat,yflat,zflat,vxflat,vyflat,vzflat)

    Wx = []; Wy = []; Wz = []

    for i in range(N**3):
        wx,wy,wz = curl(xflat[i],xflat[i],xflat[i],Vx,Vy,Vz)
        Wx.append(wx); Wy.append(wy); Wz.append(wz) 


    Xi = xi[:,:,N/2]
    Yi = yi[:,:,N/2]
    Uxi = uxi[:,:,N/2]
    Uyi = uyi[:,:,N/2]

    wxi,wyi,wzi = CIC(xflat,yflat,zflat,Wx,Wy,Wz)

    fig1 = plt.figure(figsize=(9,9))
    plt.rc('font',size=16)
    plt.rc('xtick.major',pad=5); plt.rc('xtick.minor',pad=5)
    plt.rc('ytick.major',pad=5); plt.rc('ytick.minor',pad=5)
    plt.imshow(wzi[:,:,N/2], cmap='winter' )
    plt.xlabel('$x$',fontsize=16)
    plt.ylabel('$y$',fontsize=16)
    plt.colorbar()
    plt.savefig('vorticity/vorticityat%04d'%n)
    plt.close('all')
    
    return wzi
