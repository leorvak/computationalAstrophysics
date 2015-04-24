#
import numpy as np
import matplotlib as mpl
from scipy import interpolate as intp
from scipy import integrate 

def f(x):
    return np.exp(-x)/x
    
#
#  integrate with different methods using Trapezoidal, Simpson, Gauss-Legendre, Gauss-Laguerre method
#

a= 1.0; b=100.0

N = [10,20,40,80,200,500,1000]

for n in N:
    h = (b-a)/n; x=np.arange(a,b,h); y=f(x)
    itrapz = integrate.trapz(y,x,h) 
    isimp  = integrate.simps(y,x,h)
    igauss = integrate.fixed_quad(f,a,b,n=n)[0]
    print n, itrapz, isimp, igauss
