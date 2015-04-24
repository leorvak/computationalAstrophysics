#
#  calculate matter corr. function via integral of P(k)
#  the slow brute-force version which splits integral over periods of oscillations and integrates each period via expensive
#  Romberg integral with very stringent tolerance 


import math
import numpy as np
from scipy import interpolate as intp
from scipy import integrate as integrate
from matplotlib import pylab as plt

k,Pk = np.loadtxt('matter_power_kmax10000.dat',usecols=(0,1), unpack=True)
lnk = np.log(k); lk = np.log10(k)
rcutoff = np.maximum(0.01,1.0/k[-1])

# taper the P(k) exponentially to improve convergece; 
# rcutoff should be <~0.1rmin, where rmin is smallest r in xi(r) calculation

Pk  = Pk*np.exp(-(rcutoff*k))
lPk = np.log10(Pk)

#
# setup two grids of r: finer one for small r where sharp features exist, 
# and coarser one for large r where xi(r) is expected to be featureless
#
lrsimple = 2.3 # boundary between simple integration and a (much) more difficult one
lr1 = np.arange(-1.0,lrsimple,0.01);  lr2 =np.arange(lrsimple,2.4,0.1)
lr = np.concatenate((lr1,lr2))
r = 10.0**lr

lPksp = intp.UnivariateSpline(lk,lPk,s=0.0)

lkf = np.arange(-5.0,4.0,0.0001)
kf = 10.**lkf; lnkf = np.log(kf)
intk2Pk = kf*kf*10.0**lPksp(lkf)

# will compare two functions - one using simple spline, and another using the whole shebang
xir = np.zeros_like(r)
xir2 = np.zeros_like(r)

xiout = file("xi_out.dat","w")
xiout.write('r  xi_simple_spline  xi_full nsegments used\n')

for i, rd in enumerate(r):    
    def intfunc(x):
        xd = np.exp(x)
        return xd*xd*10.0**lPksp(np.log10(xd))*np.sin(xd*rd)/rd

    # try a simple spline-based integration (good enough for small scales)
    
    intf = intk2Pk * np.sin(kf*rd)/rd
    intsp = intp.UnivariateSpline(lnkf,intf,s=0.0)
    xir[i] = intsp.integral(lnkf[0],lnkf[-1])/(2.0*math.pi**2)
    xir2[i] = xir[i]

    nmax = max(1000,int(2.0*rd/rcutoff/math.pi))
#
#   for large r, do an expensive integration breaking integral into many pieces
#   each integrating between successive zeroes of sin(kr)
#
    if ( lr[i] > lrsimple ):
        n = np.arange(1,nmax,1); kzeros = n*math.pi/rd; kzeros=np.log(kzeros)
        xir2[i] = integrate.romberg(intfunc,lnk[0],kzeros[0],tol=1.e-14)/(2.0*math.pi**2)
        for j in range(len(n)-1):
            xir2[i] += integrate.romberg(intfunc,kzeros[j],kzeros[j+1],tol=1.e-14)/(2.0*math.pi**2)
            
    print "%.4f"%rd, "%.5e"%xir[i], "%.5e"%xir2[i], nmax
    xiout.write('%.4f %.5e %.5e %d\n' % (rd, xir[i], xir2[i], nmax))

fig1 = plt.figure()

plt.rc('text', usetex=True)
plt.rc('font',size=16,**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('xtick.major',pad=10); plt.rc('xtick.minor',pad=10)
plt.rc('ytick.major',pad=10); plt.rc('ytick.minor',pad=10)

#plt.plot(lr,np.log10(np.abs(xir)),linewidth=1.5,c='b',label='$\\xi(r)$')
plt.plot(lr,np.log10(np.abs(xir2)),linewidth=1.5,c='m',label=r'$\xi(r)$')

plt.xlabel(r'\textbf{$\log_{10}(r/\rm Mpc)$}',labelpad = 5)
plt.ylabel(r'$\log_{10}(\vert\xi(r)\vert)$',labelpad = 10)
plt.title('global correlation function',fontsize=16,pad=10)
plt.legend(loc='upper right')

plt.show()
