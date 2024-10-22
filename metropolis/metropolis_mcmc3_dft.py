#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  example of the Metropolis algorithm single walker chain
#  a difficult distribution: the Rosenbrock “banana” pdf
#

import math
import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from time import time
import acor


def kPk_dft(y):
    N = len(y)
    c = np.zeros(N//2+1,complex)
    kj = np.zeros(N//2+1) 
    for k in range(N//2+1):
        kj[k] = 2.0*math.pi*k/N
        for n in range(N):
            c[k] += y[n]*np.exp(-2j*math.pi*k*n/N)
    Pkj = np.square(np.absolute(c))/N; 
    return kj, Pkj # return wavevector and power spectrum


def modelpdf(x1,x2):
    "the Rosenbrock “banana” pdf"
    return np.exp(-(100.*(x2-x1**2)**2+(1.0-x1)**2)/20.0)

# define chain parameters: N of chain entries, N of first entries to discard, step size
n = 10000; nburn=1000; nsel=1; step = 1.0
# set initial walker position
x = 0.; y = 0.
chain = []
chain.append([x,y])

# precompute a set of random numbers
nrand = n
delta = zip(rnd.uniform(-step,step,nrand),rnd.uniform(-step,step,nrand)) #random inovation, uniform proposal distribution

naccept = 0; i = 0; ntry = 0   
for nd in range(n):
    if not i%nrand:  # if ran out of random numbers generate some more
       delta = zip(rnd.uniform(-step,step,nrand),rnd.uniform(-step,step,nrand)) #random inovation, uniform proposal distribution
       i = 0  
    xtry = x + delta[i][0] # trial step
    ytry = y + delta[i][1]
    gxtry = modelpdf(xtry,ytry)
    gx = modelpdf(x,y)
    #
    # now accept the step with probability min(1.0,gxtry/gx); 
    # if not accepted walker coordinate remains the same and is also added to the chain
    #
    if gxtry > gx: 
        x = xtry; y=ytry
        naccept += 1
    else:     
        aprob = gxtry/gx # acceptance probability
        u = rnd.uniform(0,1)
        if u < aprob:
            x = xtry; y= ytry
            naccept += 1
    #
    # whatever the outcome, add current walker coordinates to the chain
    #
    chain.append([x,y])
    i += 1; ntry += 1
    
print "Generated n ",n," samples with a single walker"
print "with acceptance ratio", 1.0*naccept/ntry

#            
# plot results:
#
x = np.arange(-10.0,10.0,0.05); y = np.arange(-1.0,100,0.05)

# build grid for countours of the actual target pdf
X, Y = np.meshgrid(x,y)
# compute target pdf values at the grid points
Z = modelpdf(X,Y)

plt.rc('font', family='sans-serif', size=13)
fig=plt.figure(figsize=(13,8))
plt.subplot(221)
plt.title('Metropolis trace')
plt.plot(chain)

#
# plot the chain
#
#ax = plt.subplot(223)
ax = plt.subplot2grid((2,2),(1,0),colspan=2)
xh = np.array(zip(*chain)[0]); yh = np.array(zip(*chain)[1])
xh = xh[nburn::nsel]; yh=yh[nburn::nsel]
plt.hist2d(xh,yh, bins=100, norm=LogNorm(), normed=1)
plt.colorbar()

#
# plot theoretical contours of the target pdf
#
dlnL2= np.array([2.30, 9.21]) # contours enclosing 68.27 and 99% of the probability density
Lmax = modelpdf(0.0,0.0)
lvls = Lmax/np.exp(0.5*dlnL2)
cs=plt.contour(X,Y,Z, linewidths=(1.0,2.0), colors='black', norm = LogNorm(), levels = lvls, legend='target pdf' )

plt.title('MCMC samples vs target distribution')

labels = ['68.27%', '99%']
for i in range(len(labels)):
    cs.collections[i].set_label(labels[i])

plt.legend(loc='upper center')
plt.ylabel('y')
plt.xlabel('x')

ax = plt.subplot2grid((2,2),(0,1),colspan=1)

print "computing DFT..."
t1 = time()
#kj, Pkj = kPk_dft(xh)
t2 = time()
print "done in",t2-t1," seconds."

print "computing autocorrelation time..."
t1 = time()
tacor = acor.acor(xh,100)
t2 = time()
print "done in",t2-t1," seconds."
print "tacor=",tacor

plt.plot(np.log10(kj),np.log10(Pkj))
plt.xlabel(r'$\log_{10} k$',fontsize=12)
plt.ylabel(r'$\log_{10} P(k)$',fontsize=12)
plt.title('chain power spectrum')

plt.show()
