#
#   compute and plot stellar mass function approximated by Schechter fits
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interpolate as intp
import math

#
# prepare stellar mass function fit
#

lms = np.arange(5.0,14.,0.1) # grid of stellar masses
ms  = 10.0**lms
#
# Baldry et al. 2012 stellar mass function for small M*
#
lmstar = 10.66
phi1s = 3.96e-3; alpha1=-0.35; phi2s = 6.9e-4; alpha2=-1.57;

mstar = 10.**lmstar; mus =  ms/mstar
dnms1 = np.exp(-mus)*(phi1s*mus**alpha1 + phi2s*mus**alpha2)/mstar

#
# using Bernardi et al. 2013 double Schechter fit for large M*
#

mstarb = 0.0094e9; phisb = 1.040e-2; alphab = 1.665; betab = 0.255
phisg = 0.675e-2; mstarg = 2.7031e9; gammag = 0.296

gammanorm = math.gamma(alphab/betab)

musb = ms/mstarb; musg = ms/mstarg
dnms2 = (phisb*np.exp(-musb**betab)*musb**(alphab-1)/(mstarb)*betab/gammanorm +
         phisg*musg**(gammag-1)*np.exp(-musg)/mstarg)

#
# multiply by M* to get dn/dlnM and take maximum 
# of Baldry et al. and Bernardi et al stellar mass functions to construct the composite
#
dnms1 = dnms1*ms; dnms2 = dnms2*ms
dnms = np.maximum(dnms1,dnms2)

#
#  plot 
#
#plt.switch_backend('TkAgg')

fig1 = plt.figure()
plt.plot(lms,np.log10(dnms),linewidth=1.5,c='b',label='composite SMF')

plt.xlabel('$\\log_{10} M_*$')
plt.ylabel('$dn/d\\ln M_*$')
plt.title('composite stellar mass function')
plt.legend(loc='lower left')

plt.show()

