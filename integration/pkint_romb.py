#
import numpy as np
import matplotlib as mpl
from scipy import interpolate as intp
from scipy import integrate 
import matplotlib.pyplot as plt


fname = './test_matterpower_logintk100.dat'
k, Pk = np.loadtxt(fname,usecols=(0,1),unpack=True)

#k = np.log(k); Pk = Pk*np.exp(k)

# construct interpolated spline and ise SciPy to calculate its integral
#

sPk0 = intp.UnivariateSpline (k, Pk, s=0.0) # effectively standard interpolated spline
ssp = sPk0.integral(k[0],k[-1])
print "SciPy spline integral:", ssp

sromb = integrate.romberg(sPk0,k[0],k[-1],rtol=1.e-10,divmax=20,show=True)

frac = (sromb-ssp)/ssp
print "SciPy Romberg integral:",sromb
print "frac difference from spline integral:", frac