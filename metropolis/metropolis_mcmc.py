#!/usr/bin/env python
#
#  example of the Metropolis algorithm chain
#

import math
import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt

def modelpdf(x):
    " simple Gaussian"
    return np.exp(-x*x/2.)/np.sqrt(2*math.pi)

n = 1000000; nburn=0; nsel=1; step = 0.5
x = 0.
xchain = []
xchain.append(x)

delta = rnd.uniform(-step,step,2*n) #random trial steps

#
# a simple single chain MCMC Metropolis sampler
#
nsample = 0; i = 0  
while nsample < n:
    xtry = x + delta[i] # trial step
    gxtry = modelpdf(xtry); gx = modelpdf(x)
    if gxtry > gx: 
        x = xtry; 
    else:     
        aprob = gxtry/gx # acceptance probability
        u = rnd.uniform(0,1)
        if u < aprob:
            x = xtry
    nsample += 1
    xchain.append(x)
    i += 1
#            
# plot results:
#
x = np.arange(-6.0,6.0,0.01); y = modelpdf(x)

xnew = xchain[nburn::nsel]

plt.rc('font', family='sans-serif', size=16)
plt.figure(figsize=(6,10))
plt.subplot(211)
plt.title('Metropolis')
plt.plot(xchain)

plt.subplot(212)
plt.hist(xnew, bins=400,normed=1.0)
plt.plot(x,y,linewidth=2.5,color='r')
plt.ylabel('Frequency')
plt.xlabel('x')
#plt.legend(('target Gaussian','MCMC samples'))
plt.show()