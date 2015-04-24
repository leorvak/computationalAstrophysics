#
import numpy as np
import matplotlib as mpl
from scipy import interpolate as intp
from scipy import integrate 
import matplotlib.pyplot as plt


fname = './test_matterpower_logintk100.dat'
k, Pk = np.loadtxt(fname,usecols=(0,1),unpack=True)

#
# construct interpolated spline and ise SciPy to calculate its integral
#

sPk0 = intp.UnivariateSpline (k, Pk, s=0.0) # effectively standard interpolated spline
ssp = sPk0.integral(k[0],k[len(k)-1])
print "SciPy spline integral:", ssp

sscipytrap = integrate.cumtrapz(Pk,k)
print "SciPy trapezoidal:", sscipytrap[-1]

#
# calculate trapezoidal rule on our own
#

dxd = (np.roll(k,-1)-np.roll(k,1))
dxd[0]=k[1]-k[0]; dxd[-1]=k[-1]-k[len(k)-2]

strapz = np.sum(np.dot(dxd,Pk))*0.5
frac0 = (strapz-sscipytrap[-1])/sscipytrap[-1]
frac1 = (strapz-ssp)/ssp
print "My trapezoidal and frac. difference from SciPy trap and spline:"
print strapz, frac0, frac1

#
# scipy Simpson
#
#  even = "first" instructs it to use trap. rule on the last interval 
#     for even number of samples
#
sscipysimp = integrate.simps(Pk,k,even="first")
fracspspsimp = (sscipysimp-ssp)/ssp
print "SciPy Simpson and frac. difference from spline:"
print sscipysimp, fracspspsimp

# 
# calculate Simpson rule on our own
#
dxd = [0]*(len(k))

dx31  = (np.roll(k,-1)-np.roll(k,1))
dx32  = np.roll(k,-1)-k
dx21  = k-np.roll(k,1)

dx321 = (k-np.roll(k,2))*(2.0*k-3.0*np.roll(k,1)+np.roll(k,2))/dx21 - \
        (np.roll(k,-2)-k)*(2.0*k-3.0*np.roll(k,-1)+np.roll(k,-2))/dx32

dxd = dx31**3/(dx32*dx21)
dxd[::2]= dx321[::2]
dxd[0]  = (k[2]-k[0])*(3.0*k[1]-2.0*k[0]-k[2])/(k[1]-k[0])

if (len(dxd)%2) : 
    print "odd!"
    dxd[-1] = (k[-1]-k[len(k)-3])*(2.0*k[-1]-3.0*k[len(k)-2]+k[len(k)-3])/(k[-1]-k[len(k)-2])
    ssimp = np.sum(np.dot(dxd,Pk))/6.0
else:
    print "even!"
    dxd[len(dxd)-2] = (k[len(k)-2]-k[len(k)-4])*(2.0*k[len(k)-2]-3.0*k[len(k)-3]+k[len(k)-4])/(k[len(k)-2]-k[len(k)-3])
    dxd=dxd[:len(k)-1]; Pkd=Pk[:len(Pk)-1]
    ssimp = np.sum(np.dot(dxd,Pkd))/6.0 + 0.5*(k[-1]-k[len(k)-2])*(Pk[-1]+Pk[len(Pk)-2])
    
fracsimp = (ssimp-sscipysimp)/sscipysimp
fracsimpsp = (ssimp-ssp)/ssp
print "My Simpson and frac. difference from SciPy Simp. and spline:"
print  ssimp, fracsimp, fracsimpsp


