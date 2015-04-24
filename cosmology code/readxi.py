import numpy as np
from matplotlib import pylab as plt
    
def read_DR11_xi():
    """
    reads DR11 BOSS corr. function for LRG galaxies and corresponding covariance matrix
    returns r, xi vectors and covij = matrix lxl where l=len(r)=len(xi)
    covij[i][j] contains covariance of errors of xi(r_i) and xi(r_j)
    """
    
    r,xi = np.loadtxt('Anderson_2013_CMASSDR11_corrfunction_x0x2_postrecon.dat',usecols=(0,1), unpack=True)
    ri, rj, covij = np.loadtxt('Anderson_2013_CMASSDR11_corrfunction_cov_x0x2_postrecon.dat',usecols=(0,1,2), unpack=True)

    lmat = np.sqrt(len(ri))
    covij = np.reshape(covij,(lmat,lmat))
    return r, xi, covij


r, xi, covij = read_DR11_xi()

xir2 = xi*r*r

rm, xim = np.loadtxt('xi_out.dat',usecols=(0,1), unpack=True)

ximr2 = xim*rm*rm*2.2 # multiply by arbitrary bias 
exir2 = np.sqrt(np.diagonal(covij))*r*r

#
# plot corr. function along with a model
#


fig1 = plt.figure()

plt.rc('text', usetex=True)
plt.rc('font',size=16)
plt.rc('xtick.major',pad=5); plt.rc('xtick.minor',pad=5)
plt.rc('ytick.major',pad=5); plt.rc('ytick.minor',pad=5)

plt.xlim(40.0,200.0)
plt.errorbar(r,xir2,yerr=exir2,linestyle="None",marker="None",ecolor='b',linewidth=2,capthick=2,label=r'DR11 $\xi(r)$ re-con')
plt.scatter(r,xir2,c='b',s=70)

plt.plot(rm,ximr2,c='m',linewidth=3.0,label=r'a $\Lambda$CDM model')
#
plt.xlabel(r'$r\ (h^{-1}\rm\, Mpc)$')
plt.ylabel(r'$r^2\xi(r)\ (h^{-2}\,\rm Mpc^2)$')
plt.title('correlation function')
plt.legend(loc='upper right')
#
plt.show()
