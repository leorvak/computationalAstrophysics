#
#  example of how to use Cosmology class in cosmology.py:
#  namely, define a cosmological model, compute a power spectrum using the 
#  Einstein & Hu (1998) approximation with and without baryonic acoustic oscillation wiggles
#
import numpy as np
import cosmology


if __name__ == "__main__":

    # first a few lines to show you how to interface with routines in the cosmology module

    # define a vector of cosmological parameters:    
    my_cosmo = {'flat': True, 'H0': 72.0, 'Om0': 0.25, 'Ob0': 0.043, 'sigma8': 0.8, 'ns': 0.97}
    # set my_cosmo to be the current cosmology	
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
    print(("Omega_m = %.2f, Omega_L = %.2f" % (cosmo.Om0, cosmo.OL0)))

    # change Omega_m0 of the current cosmology
    cosmo.Om0 = 0.27; h=0.4
    print(("Omega_m = %.2f, Omega_L = %.2f" % (cosmo.Om0, cosmo.OL0)))

    # this, however, makes current cosmology non-flat, so after such change need to call
    # cosmo.checkForChangedCosmology
    cosmo.checkForChangedCosmology()
    print(("Omega_m = %.2f, Omega_L = %.2f" % (cosmo.Om0, cosmo.OL0)))
    # set redshift (could be a numpy array of redshifts; all cosmology functions can be called with numbers or numpy arrays)
    z = 0.57
    print(("At z="+str(z)+"\nAge               = " + str(cosmo.age(z))+" Gyrs"))
    print(("Comoving distance = " + str(cosmo.comovingDistance(z_max = z))+" 1/h Mpc"))
    
    lk = np.arange(-4.0,4.0,0.01); k = np.power(10.0,lk)
 
    # get full P(k) with BAO oscillations using the Eisenstein & Hu (1998) approximation
    # for our defined cosmology cosmo
    # ignore_norm = False means that it will normalize the power spectrum using current sigma8
    Pk      = cosmo.matterPowerSpectrum(k, 'eh98', ignore_norm = False)
    
    # compute smooth P(k) without baryonic wiggles
    Pksmooth = cosmo.matterPowerSpectrum(k, 'eh98smooth', ignore_norm = False)
    
    #
    #  plot the power spectra and their ratio
    #
    from socket import gethostname
    if ( gethostname()[0:6] == 'midway' ):
        plt.switch_backend('TkAgg')
    from matplotlib import pylab as plt
    
    fig1 = plt.figure(figsize=(8,12))
    kmin = 1.e-4; kmax=10.0
    #plt.rc('text', usetex=True)  # uncomment to TeX-ify the labels 
    plt.rc('font',size=16)
    plt.rc('xtick.major',pad=5); plt.rc('xtick.minor',pad=5)
    plt.rc('ytick.major',pad=5); plt.rc('ytick.minor',pad=5)

    ax1 = plt.subplot(211)
    plt.xscale('log'); plt.yscale('log')
    plt.xlim(kmin,kmax); plt.ylim(0.11,50000)
    plt.plot(k,Pk,c='m',linewidth=2.0,label=r'$P(k)$')
    plt.plot(k,Pksmooth,c='b',linewidth=2.0,label=r'$P_{\rm smooth}(k)$')
    plt.setp( ax1.get_xticklabels(), visible=False)
    plt.ylabel(r'$P(k)\ (h^{-3}\,\rm Mpc^3)$')
    plt.title('power spectra')
    plt.legend(loc='lower left')

    ax2 = plt.subplot(212, sharex=ax1)
    plt.xlim(kmin,kmax);
    plt.plot(k,Pk/Pksmooth,c='b',linewidth=2.0,label=r'ratio')
    plt.ylabel(r'$P(k)/P_{\rm smooth}(k)$')
    plt.xlabel(r'$k\ (h\rm\, Mpc^{-1})$')
    #plt.tight_layout()
    plt.subplots_adjust(hspace=0.025)
    plt.show()
