'''
PM code for cosmological initial conditions

Authors: Pablo Lemos
         Feras Aldahlawi
         
Date: 12/09/2014
'''

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

seed = 571;
rnd.seed(seed)
    
#Initial Cosmological Conditions
my_cosmo = {'flat': True, 'H0': 70.0, 'Om0': 0.274, 'Ob0': 0.0432, 'sigma8': 0.8, 'ns': 0.95}
#my_cosmo = {'flat': True, 'H0': 70.0, 'Om0': 1.0, 'Ob0': 0.0432, 'sigma8': 0.8, 'ns': 0.95}
cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
cosmo.checkForChangedCosmology()
lk = np.arange(-3.0,2.0,0.01); k = np.power(10.0,lk)

#Grid Size
N = 64; N3 = N**3
Lbox = 64.
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
         
#compute the density (rho)
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
    
#Calculates acceleration 
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
        gx = gx*-1.
        gy = gy*-1.
        gz = gz*-1.
       
        return gx, gy, gz, phi

        
#CIC for acceleration (From Kravtsov)
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

#compute power spectrum 
def computePk(rho,a,n,x,y,z):
    
    Pkm = cosmo.matterPowerSpectrum(k, 'eh98', ignore_norm = False)
    Pkm = Pkm*Dplus(atoz(a))**2
        
    rhof = fftn(rho)
    rhof[0,0,0] = 0.0

    rhof = np.square(np.absolute(rhof))/N**3
    
    #compute the power spectrum    
    bins = N
    nbin,bins2 = np.histogram(klmn,bins)
    meanbin,bins2 = np.histogram(klmn,bins,weights = rhof)
    sqbin,bins2 = np.histogram(klmn,bins,weights = rhof**2)
    
    #normalized mean    
    Pknum = meanbin/nbin
    
    sigma = np.sqrt((sqbin/nbin - Pknum*Pknum)/(nbin-1))
    
    knum = (bins2[1:] + bins2[:-1])/2
        
    plt.figure(figsize=(12,12))
    plt.title(r'Power Spectrum')
    plt.xscale('log'); plt.yscale('log')
    plt.rc('text', usetex=True)
    plt.rc('font',size=16)
    plt.rc('xtick.major',pad=5); plt.rc('xtick.minor',pad=5)
    plt.rc('ytick.major',pad=5); plt.rc('ytick.minor',pad=5)
        
    plt.ylim(1.e-5,1.e5); plt.xlim(1.e-3,1.e2)
    plt.errorbar(knum,Pknum,yerr=2.*sigma,ecolor='b',fmt='o',linewidth=1,capthick=1)
    plt.scatter(knum,Pknum,c='b',s=75,label=r'Numerical')
    plt.xlabel('$k(hMpc^{-1})$',fontsize=16)
    plt.ylabel('$P(k) (h^{-3}Mpc^{3}$',fontsize=16)
    plt.plot(k,Pkm,c='m',linewidth=2.5,label=r'EH98')
    plt.legend()
    plt.savefig('powerSpectrum/powerspectrumat%04d'%n)
    plt.close('all')
        
#Evolves particles
def evolve(x0, y0, z0, px0, py0, pz0, aini, da):    
    an = aini
    
    x = x0.copy();  y = y0.copy(); z = z0.copy()
    px = px0.copy(); py = py0.copy(); pz = pz0.copy()
    
    rhoini = np.zeros([N,N,N])
    
    rho = computeRho(x,y,z,rhoini)
    ax, ay, az, phi = updateAcceleration(rho,an)
    gx, gy, gz = gintp(x,y,z, ax, ay, az)
    
    g_array = np.array([gx])
    phi_array = np.array([phi])
    
    n = 0;
    while (an < (1.0)):
        
        an = aini + n*da
        print 'a = ',an
   
        ax, ay, az, phi = updateAcceleration(rho,an)
        gx, gy, gz = gintp(x,y,z, ax, ay, az)
        
        px +=  f(an)*gx*da
        py +=  f(an)*gy*da
        pz +=  f(an)*gz*da
        
        x += (px * da *  f(an-0.5*da +da))/((an-0.5*da +da)**2) 
        y += (py * da *  f(an-0.5*da +da))/((an-0.5*da +da)**2) 
        z += (pz * da *  f(an-0.5*da +da))/((an-0.5*da +da)**2) 
        
        rho = computeRho(x,y,z,rhoini)
                       
        xr = np.ravel(x)
        yr = np.ravel(y)
        zr = np.ravel(z)
        
        
        
        wzi = computeVorticity(x,y,z,px,py,pz,n)    
        computePk(rho,an,n,xr,yr,zr)
        
        n += 1
    
    computePk(wzi,an,9999,xr,yr,zr)
    return

################## Main  ##############
aini = ztoa(30.)
da = 0.01


particles = rnd.normal(0.,1.,size = (N,N,N)) 

#Fourier transform of the amplitudes
lambd = fftn(particles)

    
#scaled amplitudes
zini = atoz(aini)

Pk = cosmo.matterPowerSpectrum(klmn, 'eh98', ignore_norm = False)
lambd = lambd * Dplus(zini) * np.sqrt(Pk)/cosmo.H0
lambd[0,0,0] = 0.0

        
lambdx = 1.0j*(kx/klmn**2) * (1./Dplus(zini)) * lambd
lambdy = 1.0j*(ky/klmn**2) * (1./Dplus(zini)) * lambd
lambdz = 1.0j*(kz/klmn**2) * (1./Dplus(zini)) * lambd

rlambdx = np.real(ifftn(lambdx))
rlambdy = np.real(ifftn(lambdy))   
rlambdz = np.real(ifftn(lambdz))           
        
x1d = np.linspace(0.0, N-1, N)
qxi, qyi, qzi = np.meshgrid(x1d, x1d, x1d,indexing = 'ij')
       
xini = (qxi + Dplus(zini)*rlambdx + N)%N
yini = (qyi + Dplus(zini)*rlambdy + N)%N
zini = (qzi + Dplus(zini)*rlambdz + N)%N

#compute the derivative of Dplus as an approximation
Ddot = (aini - 0.5*da)*cosmo.Ez(atoz(aini - 0.5*da))*Dplus(100.0)/ztoa(100.)
           
pxini = (aini - 0.5*da)**2*Ddot*rlambdx
pyini = (aini - 0.5*da)**2*Ddot*rlambdy
pzini = (aini - 0.5*da)**2*Ddot*rlambdz

evolve(xini, yini, zini, pxini, pyini, pzini, aini, da)
