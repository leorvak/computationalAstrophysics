#
import numpy as np
import matplotlib as mpl
from scipy import interpolate as intp
from scipy import integrate 
import matplotlib.pyplot as plt


fname = './test_matterpower_logintk100.dat'
k, Pk = np.loadtxt(fname,usecols=(0,1),unpack=True)

lk = np.log(k); lPk = np.log10(Pk)

#
# construct interpolated spline and ise SciPy to calculate its integral
#

Pkf = Pk*np.exp(lk)

sPkl = intp.UnivariateSpline (lk, Pkf, s=0.0) # effectively standard interpolated spline
sspl = sPkl.integral(lk[0],lk[-1])
sPk = intp.UnivariateSpline (k, Pk, s=0.0) # effectively standard interpolated spline
ssp = sPk.integral(k[0],k[-1])

print "SciPy spline integrals (log and non-log k):", sspl, ssp

norder = 500
sgauss  = integrate.fixed_quad(sPk,k[0],k[-1],n=norder)
sgaussl = integrate.fixed_quad(sPkl,lk[0],lk[-1],n=norder) 
frac  = (sgauss[0]-ssp)/ssp
fracl = (sgaussl[0]-ssp)/ssp

print "SciPy Gauss method integral (nonlog and log k):",sgauss[0], sgaussl[0]
print "frac difference from spline integral (nonlog and log k):", frac, fracl