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


def modelpdf(x1,x2,params):
    "not so simple Gaussian: 2d Gaussian with non-zero correlation coefficient r"
    x1m = params[0]; x2m = params[1]
    sig1 = params[2]; sig2 = params[3]; r = params[4]
    r2 = r*r
    return np.exp(-0.5*(((x1-x1m)/sig1)**2+((x2-x2m)/sig2)**2-2.0*r*(x1-x1m)*(x2-x2m)/(sig1*sig2))/(1.0-r2))/(2*math.pi*sig1*sig2)/np.sqrt(1.0-r2)

def modelpdf2(x1,x2,params):
    "the Rosenbrock “banana” pdf"
    return np.exp(-(100.*(x2-x1**2)**2+(1.0-x1)**2)/20.0)

x1 = 0.0; x2 = 0.0; s1 = 1.0; s2 = 1.0; r12=0.7
params=np.array(5); params=(x1,x2,s1,s2,r12)

# define chain parameters: N of chain entries, N of first entries to discard, step size
n = 1000000; nburn=1000; nsel=1; step = 1.0
# set initial walker position
x = x1; y = x2
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
    gxtry = modelpdf(xtry,ytry,params)
    gx = modelpdf(x,y,params)
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

xh = np.array(zip(*chain)[0]); yh = np.array(zip(*chain)[1])
xh = xh[nburn::nsel]; yh=yh[nburn::nsel]

print "computing autocorrelation time..."
t1 = time()
tacor = acor.acor(xh,100)
t2 = time()
print "done in",t2-t1," seconds."
print "tacor=",tacor
