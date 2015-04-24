# -*- coding: utf-8 -*-
#  a difficult distribution: the Rosenbrock “banana” pdf
#  sampled with the Goodman & Weare 2010 affine invariant sampling algorithm
#  splitting number of walkers into two sub-samples to prepare for parallelization
#  the code is written for an arbitrary modelpdf function which takes x[nparams] vector of parameters
#  the chain samples nparams parameters with specified number of walkers nwalkers
#
import math
import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#import acor
from time import time
from time import sleep

def modelpdf(x):
    "not so simple Gaussian: 2d Gaussian with non-zero correlation coefficient r"
    sig1 = 1.0; sig2 = 1.0; r=0.95
    r2 = r*r
    x1 = x[0]; x2=x[1]
    return np.exp(-0.5*((x1/sig1)**2+(x2/sig2)**2-2.0*r*x1*x2/(sig1*sig2))/(1.0-r2))/(2*np.pi*sig1*sig2)/np.sqrt(1.0-r2)

def modelpdf2(x):
    "the Rosenbrock “banana” pdf not so simple Gaussian: 2d Gaussian with non-zero correlation coefficient r"
    x1 = x[0]; x2 = x[1]
    return np.exp(-(100.*(x2-x1**2)**2+(1.0-x1)**2)/20.0)

rnd.seed(651)

nparams = 2
n = 100; nburn=0; nsel=1; step = 0.1
ap = 2.0; api = 1.0/ap; asqri=1.0/np.sqrt(ap); afact=(ap-1.0)
nwalkers = 100
#
# distribute initial positions of walkers in an isotropic Gaussian around the initial point
#
x0 = np.zeros(nparams)

x = np.zeros([2,nwalkers/2,nparams])

for i in range(nparams):
    x[:,:,i] = np.reshape(rnd.normal(x0[i],step,nwalkers),(2,nwalkers/2))

chain = []
nRval = 100 # how often to compute R
Rval = []

naccept = 0; ntry = 0; nchain = 0
mw = np.zeros((nwalkers,nparams)); sw = np.zeros((nwalkers,nparams))
m = np.zeros(nparams)
Wgr = np.zeros(nparams); Bgr = np.zeros(nparams); Rgr=np.zeros(nparams)
converged = False;

while not converged:
    for kd in range(2):
        k = abs(kd-1)
        for i in range(nwalkers/2):
            zf= rnd.rand()   # the next few steps implement Goodman & Weare sampling algorithm
            zr = (1.0+zf*afact)**2*api
            j = rnd.randint(nwalkers/2)
            xtry = x[kd,j,:] + zr*(x[k,i,:]-x[kd,j,:])
            gxtry = modelpdf(xtry)
            gx = modelpdf(x[k,i,:])
            gx = np.max(np.abs(gx),1.e-50)
            aprob = zr*gxtry/gx
            if aprob >= 1.0:
                x[k,i,:] = xtry
                naccept += 1
            else:
                u = rnd.uniform(0,1)
                if u < aprob:
                    x[k,i,:] = xtry
                    naccept += 1
            if nchain >= nburn:
                if ( nchain == 199 ):
                    print x[k,i,:]
                chain.append(np.array(x[k,i,:]))
                mw[k*nwalkers/2+i,:] += x[k,i,:]
                sw[k*nwalkers/2+i,:] += x[k,i,:]**2
            ntry += 1
        #if ( nchain == 201 ):
        #    break #sleep(20)
    nchain += 1

    if ( nchain >= nburn and nchain > 2 and nchain%nRval == 0):
        # use Gelman & Rubin convergence instead of the flaky corr. time
        mwc = mw/(nchain-1.0)
        swc = sw/(nchain-1.0)-np.power(mwc,2)

        for i in range(nparams):
            # within chain variance
            Wgr[i] = np.sum(swc[:,i])/nwalkers
            # mean of the means over Nwalkers
            m[i] = np.sum(mwc[:,i])/nwalkers
            # between chain variance
            Bgr[i] = nchain*np.sum(np.power(mwc[:,i]-m[i],2))/(nwalkers-1.0)
            # Gelman-Rubin R factor
            Rgr[i] = (1.0 - 1.0/nchain + Bgr[i]/Wgr[i]/nchain)*(nwalkers+1.0)/nwalkers - (nchain-1.0)/(nchain*nwalkers)
        Rval.append(Rgr-1.0)
        print "nchain=",nchain
        print "R values for parameters:", Rgr
        print mwc, m
        if np.max(Rgr-1.0) < 0.05: converged = True
#        if ( nchain > 300 ): converged = True
        
print "Generated ",ntry," samples using", nwalkers," walkers"
print "with step acceptance ratio of", 1.0*naccept/ntry

xh = zip(*chain)[0]; yh=zip(*chain)[1]

#
# plot the results
#
from socket import gethostname

if ( gethostname()[0:6] == 'midway' ):
    plt.switch_backend('TkAgg')

x = np.arange(np.min(xh)-1.0,np.max(xh)+1,0.05); y = np.arange(np.min(yh)-1.0,np.max(yh)+1,0.05)

X,Y = np.meshgrid(x,y)
Z = modelpdf((X,Y))

plt.rc('font', family='sans-serif', size=16)
fig=plt.figure(figsize=(10,15))
plt.subplot(211)
plt.title('Goodman & Weare 2010 sampler')
#plt.plot(chain)
plt.yscale('log')

plt.plot(Rval)
plt.ylabel(r'$R_{GR}$')
plt.xlabel(r'iteration/nRval')

ax = plt.subplot(212)
plt.xlim(np.min(xh)-1.0,np.max(xh)+1.0)
plt.ylim(np.min(yh)-1.0,np.max(yh)+1.0)
plt.hist2d(xh,yh, bins=100, norm=LogNorm(), normed=1)
plt.colorbar()

dlnL2= np.array([2.30, 9.21]) # contours enclosing 68.27 and 99% of the probability density
Lmax = modelpdf(x0)
lvls = Lmax/np.exp(0.5*dlnL2)
cs=plt.contour(X,Y,Z, linewidths=(1.0,2.0), colors='black', norm = LogNorm(), levels = lvls, legend='target pdf' )

plt.title('MCMC samples vs target distribution')

labels = ['68.27%', '99%']
for i in range(len(labels)):
    cs.collections[i].set_label(labels[i])

plt.legend(loc='upper center')
plt.ylabel('y')
plt.xlabel('x')

plt.show()

