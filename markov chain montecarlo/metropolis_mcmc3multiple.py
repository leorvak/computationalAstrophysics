#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  example of the Metropolis algorithm chain
#  a difficult distribution: the Rosenbrock “banana” pdf
#  example of multiple chains 
#

import math
import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def modelpdf(x1,x2):
    "the Rosenbrock “banana” pdf"
    return np.exp(-(100.*(x2-x1**2)**2+(1.0-x1)**2)/20.0)

nwalkers = 100
n = 1000; nburn=100; nsel=1; step = 1.0
x0 = y0 = 0.0;
x = rnd.normal(x0,step,nwalkers); y = rnd.normal(y0,step,nwalkers)
chain = []

# precompute a set of random numbers
nrand = n*nwalkers
delta = zip(rnd.uniform(-step,step,nrand),rnd.uniform(-step,step,nrand)) #random inovation, uniform proposal distribution

naccept = 0; i = 0; ntry = 0   
for nd in range(n):
    for i in range(nwalkers):
        xtry = x[i] + delta[ntry][0] # trial step
        ytry = y[i] + delta[ntry][1]
        gxtry = modelpdf(xtry,ytry)
        gx = modelpdf(x[i],y[i])
        if gxtry > gx: 
            x[i] = xtry; y[i]=ytry
            naccept += 1
        else:     
            aprob = gxtry/gx # acceptance probability
            u = rnd.uniform(0,1)
            if u < aprob:
                x[i] = xtry; y[i]= ytry
                naccept += 1
        if nd > nburn and (not nd%nsel) : # start the chain only after burn in
            chain.append([x[i],y[i]])
        ntry += 1
    
print "Generated n ",n*nwalkers," samples using", nwalkers," walkers"
print "with acceptance ratio", 1.0*naccept/ntry

#            
# plot results:
#
x = np.arange(-10.0,10.0,0.05); y = np.arange(-1.0,100,0.05)

X, Y = np.meshgrid(x,y)
Z = modelpdf(X,Y)


plt.rc('font', family='sans-serif', size=16)
fig=plt.figure(figsize=(10,15))
plt.subplot(211)
plt.title('Metropolis')
plt.plot(chain)

ax = plt.subplot(212)
xh = zip(*chain)[0]; yh = zip(*chain)[1]
plt.hist2d(xh,yh, bins=60, norm=LogNorm(), normed=1)
plt.colorbar()

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


plt.show()
