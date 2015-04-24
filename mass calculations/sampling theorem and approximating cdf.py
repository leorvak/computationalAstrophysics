
#  sample dn/dlnM using the sampling theorem and approximating cdf using spline
#  it uses Sheth et al. halo mass function to calculate mass fraction of total mass
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interpolate as intp
from scipy import integrate
import math

#
#  read power spectrum
#
fname = 'matter_power_kmax10000.dat'
k, Pk = np.loadtxt(fname,usecols=(0,1),unpack=True)

#
# set relevant cosmological parameters
#
h = 0.7; Omega_m = 0.276; rho_mean = 2.77e11*h*h*Omega_m # in Msun/Mpc^3
#
# set a desired grid of masses and corresponding radii
#
lM = np.arange(9.0,15.4,0.1) # mass range relevant for luminous galaxies and clusters
M = 10.0**lM; R = (3.0*M/(4.0*math.pi*rho_mean))**(1.0/3.0)

# number of random samples of the pdf
nsamples = 1e7

# check if the mass limits and k limits are appropriate (see e.g. Murray et al. arxiv/1306.6721)
if not ((k[0]*R[-1]<0.1) and (k[-1]*R[0]>3.0)):
    raise ValueError("***WARNING! limits on k and R(M) will result in accurate sigma!***")

def W2(k,R):
    kR = k*R
    return (3.0*(np.sin(kR)-kR*np.cos(kR))/(kR**3))**2

def dW2dM(k,R):
    kR = k*R
    return (np.sin(kR)-kR*np.cos(kR))*(np.sin(kR)*(1.0-3.0/(kR**2))+3.0*np.cos(kR)/kR)

sig = np.zeros_like(M)
factor1 = 0.5/math.pi**2

for i, md in enumerate(M):
    sfunc = Pk*W2(k,R[i])*k*k
    sig[i] = np.sqrt(factor1*integrate.simps(sfunc,k))

#
# now compute dln(sigma)/dlnM 
#
dsdm = np.zeros_like(M)
factor2 = 1.5/math.pi**2

for i, md in enumerate(M):
    sfunc = Pk*dW2dM(k,R[i])/(k**2)
    spl = intp.UnivariateSpline(k, sfunc, s=0.0)
    dsdm[i] = factor2*spl.integral(k[0],np.inf)/sig[i]**2/R[i]**4

lsig = np.log(sig)

#
# renormalize sigma(M) to a desired sigma8
#
#
sR = intp.UnivariateSpline(R, sig, s=0.0)
R8 = 8.0/h; sig8 = sR(R8)
sig8new = 0.8
print "sigratio =", sig8new/sig8
sig = sig*sig8new/sig8

#
# mass function
#
def f_SMT(nu):
    nup2 = (0.840833*nu)**2       
    return 0.644*(1.0+1.0/nup2**0.3)*np.sqrt(nup2*0.5/math.pi)*np.exp(-0.5*nup2)      

# define peak height 
delc = 1.69; nu = delc/sig
#
# compute mass-function in the Sheth et al. 2001 approximations
#

dndlnM_SMT = rho_mean/M*abs(dsdm)*nu*f_SMT(nu)

# function to integrate for mass fractions in ln(nu)
lnM = np.log(M)

slmf = intp.UnivariateSpline(lnM,dndlnM_SMT,s=0.0)
    
# normalization integral
hi1 = slmf.integral(lnM[0],lnM[-1])

#
# compute CDF over the specified mass interval
#
hfrac = np.zeros_like(lnM); hi2 = np.zeros_like(lnM)
for i, lnMd in enumerate(lnM):
    hi2[i] = slmf.integral(lnM[0],lnMd)
    hfrac[i] = hi2[i]/hi1
    #print lM[i], hfrac[i]
#
# spline the pdf and log M
#
splcdf = intp.UnivariateSpline(hfrac,lM,s=0.0)
# sample cdf with uniform numbers in [0,1]
xrand = np.random.uniform(0.0,1.0,nsamples)
# invert cdf spline to get random numbers distributed as dn/dlnM
xsample = splcdf(xrand)

#  plot 
#

fig1 = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font',size=16,**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('xtick.major',pad=10); plt.rc('xtick.minor',pad=10)
plt.rc('ytick.major',pad=10); plt.rc('ytick.minor',pad=10)
plt.xlim(9.0,15.2), plt.yscale('log')
plt.hist(xsample, bins=60,normed=1.0)
plt.plot(lM,dndlnM_SMT/hi1/np.log10(np.exp(1.0)),linewidth=2.5,c='magenta',label='target pdf')

plt.xlabel(r'$\log_{10} M$')
plt.ylabel(r'$dn/d\log_{10} M$')
plt.title(r'sampled $dn/d\log_{10} M$')
plt.legend(loc='upper right')

plt.show()

