#
#  calculate mass functions in the Press-Schechter and Sheth et al. 2001 approximations
#
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interpolate as intp
from scipy import integrate
import math
from socket import gethostname

if ( gethostname()[0:6] == 'midway' ):
    plt.switch_backend('TkAgg')

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
lM = np.arange(1.0,16,0.1)
M = 10.0**lM; R = (3.0*M/(4.0*math.pi*rho_mean))**(1.0/3.0)
#
# check if the mass limits and k limits are appropriate (see e.g. Murray et al. arxiv/1306.6721)
#
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
# renormalize sigma(M) to a desired sigma8
#
#
sR = intp.UnivariateSpline(R, sig, s=0.0)
R8 = 8.0/h; sig8 = sR(R8)
sig8new = 0.8
print "sigratio =", sig8new/sig8
sig = sig*sig8new/sig8

#
# now compute dln(sigma)/dlnM 
#

dsdm = np.zeros_like(M)
factor2 = 1.5/math.pi**2

for i, md in enumerate(M):
    sfunc = Pk*dW2dM(k,R[i])/(k**2)
    spl = intp.UnivariateSpline(k, sfunc, s=0.0)
    dsdm[i] = factor2*spl.integral(k[0],np.inf)/sig[i]**2/R[i]**4

lsig = np.log(sig); logm = np.log(M);

#
# mass function
#
def f_PS(nu):
    return np.sqrt(2.0/math.pi)*np.exp(-0.5*nu**2)

def f_SMT(nu):
    nup2 = (0.840833*nu)**2       
    return 0.644*(1.0+1.0/nup2**0.3)*np.sqrt(nup2*0.5/math.pi)*np.exp(-0.5*nup2)      

# define peak height 
delc = 1.69; nu = delc/sig
#
# compute mass-functions in the Press-Schechter 1974 and Sheth et al. 2001 approximations
#
dndlnM_PS = rho_mean/M*abs(dsdm)*nu*f_PS(nu)
dndlnM_SMT = rho_mean/M*abs(dsdm)*nu*f_SMT(nu)

#
#  plot 
#

fig1 = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font',size=16,**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('xtick.major',pad=10); plt.rc('xtick.minor',pad=10)
plt.rc('ytick.major',pad=10); plt.rc('ytick.minor',pad=10)
plt.xscale('log'); plt.yscale('log')

plt.xlim(1.e7,1.e16); plt.ylim(1.e-11, 50.0)
plt.plot(M,dndlnM_PS,linewidth=1.5,c='r',linestyle='--',label='Press-Schechter')
plt.plot(M,dndlnM_SMT,linewidth=1.5,c='b',label='Sheth et al. 2001')

plt.xlabel(r'$M\ (M_{\odot})$')
plt.ylabel(r'$dn/d\ln M\ (Mpc^{-3})$ ')
plt.title('Press-Schecter and Sheth et al. 2001 approximations')
plt.legend(loc='lower left')

plt.show()

