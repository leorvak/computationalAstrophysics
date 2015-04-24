#
#  calculate 2nd moments of the power spectrum for the TH and Gaussian filters 
#

import math
import numpy as np
from scipy import interpolate as intp
from scipy import integrate as integrate
from matplotlib import pylab as plt
from socket import gethostname

if ( gethostname()[0:6] == 'midway' ):
    plt.switch_backend('TkAgg')

k,Pk = np.loadtxt('matter_power_kmax10000.dat',usecols=(0,1), unpack=True)
lnk = np.log(k); lk = np.log10(k)

#
# set relevant cosmological parameters
#
h = 0.7; Omega_m = 0.276; rho_mean = 2.77e11*h*h*Omega_m # in Msun/Mpc^3

lr = np.arange(-1.0,2.5,0.01)
r = 10.0**lr

#
# set a desired grid of masses and corresponding radii
#
M = 4.0*math.pi*r**3*rho_mean/3.0; lM = np.log10(M)

# intial P(k) cutoff scale
fcut = 0.001
rcutoff = np.maximum(fcut*r[0],1.0/k[-1])
#rcutoff = 0.0

# taper the P(k) exponentially to improve convergece; 
# rcutoff should be <~0.001rmin, where rmin is smallest r in xi(r) calculation

Pkc  = Pk*np.exp(-(rcutoff*k))
lPk = np.log10(Pkc)


lPksp = intp.UnivariateSpline(lk,lPk,s=0.0)

rd = 100.0
lkf = np.arange(-5.0,4.0,0.0001)
kf = 10.**lkf; lnkf = np.log(kf)
intk2Pk = kf*10.0**lPksp(lkf)
kfrd = kf*rd
intf = intk2Pk * (np.sin(kfrd)-kfrd*np.cos(kfrd))/rd**3
intsp = intp.UnivariateSpline(lnkf,intf,s=0.0)


# will compare two functions - one using simple spline, and another using the whole shebang
xir = np.zeros_like(r)
xir2 = np.zeros_like(r)

xiout = file("s2_out.dat","w")
xiout.write('r  s2_simple_spline  s2_full nsegments used\n')

for i, rd in enumerate(r):
    # increase the cutoff scale
    rcutoff = fcut*rd
    Pkc  = np.maximum(1.e-30*Pk[-1],Pk*np.exp(-(rcutoff*k)))
    lPk = np.log10(Pkc)

    lPksp = intp.UnivariateSpline(lk,lPk,s=0.0)

    intk2Pk = kf**7*10.0**lPksp(lkf)            
    kfrd = kf*rd
    intf = intk2Pk * np.exp(-kfrd**2)# ((np.sin(kfrd)-kfrd*np.cos(kfrd))/rd**3)**2
    intsp = intp.UnivariateSpline(lnkf,intf,s=0.0)

    def intfunc(x):
        xd = np.exp(x)
        xrd = xd*rd
        return xd*10.0**lPksp(np.log10(xd))*((np.sin(xrd)-xrd*np.cos(xrd))/rd**3)**2

    # try a simple spline-based integration (good enough for small scales)
    
    xir[i] = intsp.integral(lnkf[0],(lnkf[-1]))/(2.0*math.pi**2)
    
#
#   for large r, do an expensive integration breaking integral into many pieces
#   each integrating between successive zeroes of sin(kr)
#
    xir2[i] = integrate.romberg(intfunc,lnk[0],lnk[-1],tol=1.e-14)/(2.0*math.pi**2)
            
    print "%.4f"%rd, "%.5e"%xir[i], "%.5e"%xir2[i]
    xiout.write('%.4f %.5e %.5e \n' % (rd, xir[i], xir2[i]))

fig1 = plt.figure()

plt.rc('text', usetex=True)
plt.rc('font',size=16,**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('xtick.major',pad=10); plt.rc('xtick.minor',pad=10)
plt.rc('ytick.major',pad=10); plt.rc('ytick.minor',pad=10)

#rfull,xifull = np.loadtxt('../s2_out.dat',usecols=(0,1),unpack=True)

plt.plot(lr,np.log10(np.abs(xir)),linewidth=1.5,c='b',label=r'$\sigma_2^2(r)$ Gaussian (spline)')
plt.plot(lr,np.log10(np.abs(xir2)),linewidth=1.5,c='m',label=r'$\sigma_2^2(r)$ TH (Romberg)')
plt.ylim(-10.0,8.0)
plt.xlim(-0.99,2.5)
#plt.xlim(8.0,16.0); 
plt.xlabel(r'\textbf{$\log_{10}(R/\rm Mpc)$}',labelpad = 5)
plt.ylabel(r'$\log_{10}(\sigma^2_2(R))$',labelpad = 10)
plt.title('power spectrum moment',fontsize=16)
plt.legend(loc='upper right')

plt.show()
