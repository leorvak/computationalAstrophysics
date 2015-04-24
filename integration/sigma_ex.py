#
import math
import time
import numpy as np
import matplotlib as mpl
from scipy import interpolate as intp
from scipy import integrate 
import matplotlib.pyplot as plt


fname = './test_matterpower_logintk100.dat'
k, Pk = np.loadtxt(fname,usecols=(0,1),unpack=True)

h = 0.7; Omega_m = 0.276; rho_mean = 2.77e11*h*h*Omega_m # in Msun/Mpc^3

lM = np.arange(8,16,0.1)
M = 10.0**lM; R = (3.0*M/(4.0*math.pi*rho_mean))**(1.0/3.0)

#
# construct interpolated spline and ise SciPy to calculate its integral
#

def W2(k,R):
    kR = k*R
    return 3.0*(np.sin(kR)-kR*np.cos(kR))/(kR**3)


sig = [0]*len(M)
factor1 = 0.5/math.pi**2

t0 = time.time()

sPk = intp.UnivariateSpline(k, Pk, s=0.0) # effectively standard interpolated spline

for i, md in enumerate(M):
    sfunc = sPk(k)*W2(k,R[i])*k*k
    sig[i] = np.sqrt(factor1*integrate.simps(sfunc,k))

t1 = time.time()

print "sigma1=", sig[-1]

print "time via direct simpsons:", t1-t0

t2 = time.time()

for i, md in enumerate(M):
    sfunc = Pk*W2(k,R[i])*k*k
    spl = intp.UnivariateSpline(k, sfunc, s=0.0)
    sig[i] = np.sqrt(factor1*spl.integral(k[0],k[-1]))

t3 = time.time()

print "time via splines:", t3-t2

print "sigma2 =",sig[-1]

