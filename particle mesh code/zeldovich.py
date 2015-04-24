import numpy as np
import cosmology
from scipy import constants
from scipy import special
from scipy import interpolate as intp
from scipy import integrate as integrate
import math
from matplotlib import pylab as plt
from numpy import random as rnd
import time
from numpy.fft import fftn, ifftn, fftfreq
import matplotlib.animation as animation
from vort import computeVorticity

    
#Initial Cosmological Conditions
my_cosmo = {'flat': True, 'H0': 70.0, 'Om0': 0.274, 'Ob0': 0.0432, 'sigma8': 0.8, 'ns': 0.95}
#my_cosmo = {'flat': True, 'H0': 70.0, 'Om0': 1.0, 'Ob0': 0.0432, 'sigma8': 0.8, 'ns': 0.95}
cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
cosmo.checkForChangedCosmology()
lk = np.arange(-3.0,2.0,0.01); k = np.power(10.0,lk)

#Grid Size
N = 32
Lbox = 32.
Np = N**3
mp = 1.

#FFT variables (k)
G1d = fftfreq(N) * 2.0*np.pi
kx, ky, kz = np.meshgrid(G1d, G1d, G1d, indexing = 'ij')
klmn = (1.0*N/Lbox)*np.sqrt(kx**2 + ky**2 + kz**2)
klmn[0,0,0] = np.pi * 2.0 /Lbox


def ztoa(z):
    return 1./(1.+z)
    

def atoz(a):
    return (1.0/a) - 1 
    

def dzda(a):
    return -1.0/(a**2)
    

def f(a):
    return 1./np.sqrt((1./a)*(cosmo.Om0 + cosmo.OL0*a**3 + cosmo.Ok0*a))
    
def Dplus(z):
    return cosmo.growthFactor(z)

#Zeldovich plane wave
def zeldovich(aini,da):  

        zarray = np.linspace(0.,100.,1000)
        sDplus = intp.UnivariateSpline(zarray,Dplus(zarray),s=0)        
    
        zini = atoz(aini)
        zcross = atoz(10*aini)

        q1d = np.linspace(0.0, (N-1.)/N, N)*Lbox
        qxi, qyi, qzi = np.meshgrid(q1d, q1d, q1d,indexing = 'ij')
               
        z= atoz(aini - 0.5*da)
        Ddot = sDplus(z,1)*dzda(aini - 0.5*da)*cosmo.Ez(z)*ztoa(z)        
        
        k = 2.*np.pi/Lbox
        A = 1./Dplus(zcross)/k

        xn = qxi + Dplus(zini)*A*np.sin(k*qxi)
        yn = qyi + Dplus(zini)*A*np.sin(k*qxi)
        zn = qzi + Dplus(zini)*A*np.sin(k*qxi)

        px = (aini - 0.5*da)**2*Ddot*A*np.sin(k*qxi)
        py = px
        pz = px
        
        phi = (1.5*cosmo.Om0/aini)*(((qxi**2 - xn**2)/2.)+((Dplus(zini)*A/k)*((k*qxi*np.sin(k*qxi))+np.cos(k*qxi)-1)))
        gx =  (1.5*cosmo.Om0/aini)*Dplus(zini)*A*np.sin(k*qxi)

        return xn,yn,zn,px,py,pz, phi, gx

         
#Particle cell density interpolation
def computeRho(x,y,z,rho1):
    
        rho = rho1.copy()
        
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        
        ip = np.int16(x); jp = np.int16(y); kp = np.int16(z)
        dx = x - ip; dy = y - jp; dz = z - kp

        for i in range(len(x)):  
            ix = ip[i]; iy = jp[i]; iz = kp[i]
            dxi = dx[i]; dyi = dy[i]; dzi = dz[i]
            txi = 1.0-dxi; tyi = 1.0-dyi; tzi = 1.0-dzi
            
            #boudary condition
            ix = ix%N; iy = iy%N; iz = iz%N
            ixp1 = (ix+1)%N; iyp1 = (iy+1)%N; izp1 = (iz+1)%N;
            
            rho[ix,iy,iz] = rho[ix,iy,iz] + mp*txi*tyi*tzi;
            rho[ix,iyp1,iz] = rho[ix,iyp1,iz] + mp*txi*dyi*tzi;
            rho[ix,iy,izp1] = rho[ix,iy,izp1] + mp*txi*tyi*dzi;
            rho[ix,iyp1,izp1] = rho[ix,iyp1,izp1] + mp*txi*dyi*dzi;
            rho[ixp1,iy,iz] = rho[ixp1,iy,iz] + mp*dxi*tyi*tzi;
            rho[ixp1,iyp1,iz] = rho[ixp1,iyp1,iz] + mp*dxi*dyi*tzi;
            rho[ixp1,iy,izp1] = rho[ixp1,iy,izp1] + mp*dxi*tyi*dzi;
            rho[ixp1,iyp1,izp1] = rho[ixp1,iyp1,izp1] + mp*dxi*dyi*dzi;
             
        return rho
    
#Calculates acceleration field using Discretized difference method.
def updateAcceleration(rho,a):

        #Uses initial overdensity field 
        rhof = fftn(rho)
        G1d = fftfreq(N) * np.pi
        kx, ky, kz = np.meshgrid(G1d, G1d, G1d,indexing = 'ij')

        #Calculate acceleration from Green's function
        G2 = (np.sin(kx)**2+np.sin(ky)**2+np.sin(kz)**2)
        G2[0, 0, 0] = 1.0  # omit the G=0 term
        #tmp = -0.25*rhof / (G2) 
        
        #code units conversion
        tmp = -((3.*cosmo.Om0)/(8.*a))*rhof / (G2)
        tmp[0, 0, 0] = 0  # omit the G=0 term
        
        phi = np.real(ifftn(tmp))

        gx,gy,gz = np.gradient(phi)
        gx = gx*-1
        gy = gy*-1
        gz = gz*-1
       
        return gx, gy, gz, phi

        
#CIC for acceleration
def gintp(x,y,z,gxp, gyp, gzp):
    """ 
    interpolate acceleration components gxp, gyp, gzp onto particle locations
    """
    ip = np.int16(x); jp = np.int16(y); kp = np.int16(z)
    dx = x - ip; dy = y - jp; dz = z - kp

    gp = np.zeros_like(dx)
    
    gpx = np.zeros_like(dx)
    gpy = np.zeros_like(dy)
    gpz = np.zeros_like(dz)

    for i in range(len(x)):
        ix = ip[i]; iy = jp[i]; iz = kp[i]
        dxi = dx[i]; dyi = dy[i]; dzi = dz[i]
        txi = 1.0-dxi; tyi = 1.0-dyi; tzi = 1.0-dzi 

        # enforce periodic boundary
        ix = (ix)%N; iy = (iy)%N; iz = (iz)%N;
        ixp1 = (ix+1)%N; iyp1 = (iy+1)%N; izp1 = (iz+1)%N;

        gxpd = txi*tyi*tzi*gxp[ix,iy,iz] + dxi*tyi*tzi*gxp[ixp1,iy,iz] + txi*dyi*tzi*gxp[ix,iyp1,iz] + \
                dxi*dyi*tzi*gxp[ixp1,iyp1,iz] + txi*tyi*dzi*gxp[ix,iy,izp1] + dxi*tyi*dzi*gxp[ixp1,iy,izp1] + \
                txi*dyi*dzi*gxp[ix,iyp1,izp1] + dxi*dyi*dzi*gxp[ixp1,iyp1,izp1]

        gypd = txi*tyi*tzi*gyp[ix,iy,iz] + dxi*tyi*tzi*gyp[ixp1,iy,iz] + txi*dyi*tzi*gyp[ix,iyp1,iz] + \
                dxi*dyi*tzi*gyp[ixp1,iyp1,iz] + txi*tyi*dzi*gyp[ix,iy,izp1] + dxi*tyi*dzi*gyp[ixp1,iy,izp1] + \
                txi*dyi*dzi*gyp[ix,iyp1,izp1] + dxi*dyi*dzi*gyp[ixp1,iyp1,izp1]

        gzpd = txi*tyi*tzi*gzp[ix,iy,iz] + dxi*tyi*tzi*gzp[ixp1,iy,iz] + txi*dyi*tzi*gzp[ix,iyp1,iz] + \
                dxi*dyi*tzi*gzp[ixp1,iyp1,iz] + txi*tyi*dzi*gzp[ix,iy,izp1] + dxi*tyi*dzi*gzp[ixp1,iy,izp1] + \
                txi*dyi*dzi*gzp[ix,iyp1,izp1] + dxi*dyi*dzi*gzp[ixp1,iyp1,izp1]

        gpx[i] = gxpd; gpy[i] = gypd; gpz[i] = gzpd;

    return gpx, gpy, gpz
        
#Evolves particles
def evolve(x0, y0, z0, px0, py0, pz0, aini, da):
    
    an = aini

    x = x0.copy();  y = y0.copy(); z = z0.copy()
    px = px0.copy(); py = py0.copy(); pz = pz0.copy()
    
    rhoi = np.zeros([N,N,N])
    
    rho = computeRho(x,y,z,rhoi)
    ax, ay, az, phi = updateAcceleration(rho,an)
    gx, gy, gz = gintp(x,y,z, ax, ay, az)
    
    g_array = np.array([gx])
    phi_array = np.array([phi])
    
    n=0
    while (an+da < (10*aini)):
        
        ax, ay, az, phi = updateAcceleration(rho,an)
        gx, gy, gz = gintp(x,y,z, ax, ay, az)
        
        px +=  f(an)*gx*da
        py +=  f(an)*gy*da
        pz +=  f(an)*gz*da
        
        x += (px * da *  f(an-0.5*da +da))/((an-0.5*da +da)**2) 
        y += (py * da *  f(an-0.5*da +da))/((an-0.5*da +da)**2) 
        z += (pz * da *  f(an-0.5*da +da))/((an-0.5*da +da)**2)
        
        xa,ya,za,pxa,pya,pza, phia, gxa = zeldovich(an,da)
        
        xp = x_array[i,:,N/2,N/2]
        px = px_array[i,:,N/2,N/2]
        
        xp_analytic = xa[:,N/2,N/2]
        px_analytic = pxa[:,N/2,N/2]
       
    
        plt.figure(figsize=(12,12))
        plt.title(r'Zeldovich')
        plt.rc('text', usetex=True)
        plt.rc('font',size=16)
        plt.rc('xtick.major',pad=5); plt.rc('xtick.minor',pad=5)
        plt.rc('ytick.major',pad=5); plt.rc('ytick.minor',pad=5)
        
        plt.ylim(-3,3); plt.xlim(0,N)
        plt.plot(xp,px,'bo',label=r'Numerical')
        plt.xlabel('$P_x$',fontsize=16)
        plt.ylabel('$x$',fontsize=16)
        plt.plot(xp_analytic,px_analytic,c='m',linewidth=2.5,label=r'Analytical')
        plt.legend()
        plt.savefig('zeldovich/zeldovichat%04d'%i)
        plt.close('all') 
        
        rho = computeRho(x,y,z,rhoi)
        
        an += da
        n += 1
        
    return


##################Test##############
aini = ztoa(30.)
da = 0.01

xini,yini,zini,pxini,pyini,pzini, phiini, gxini = zeldovich(aini,da)

evolve(xini, yini, zini, pxini, pyini, pzini, aini, da)

