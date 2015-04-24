#!/usr/bin/env python
#
#
#  example of the Metropolis algorithm single chain
#  a more "difficult" distribution - bivariate Gaussian of 2 correlated variables
#  with a simple annealing in the end to estimate the maximum of the distribution
#

import math
import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def modelpdf(x1,x2, sig1, sig2, r):
    "not so simple Gaussian: 2d Gaussian with non-zero correlation coefficient r"
    r2 = r*r
    return np.exp(-0.5*((x1/sig1)**2+(x2/sig2)**2-2.0*r*x1*x2/(sig1*sig2))/(1.0-r2))/(2*math.pi*sig1*sig2)/np.sqrt(1.0-r2)

s1 = 1.0; s2 = 1.0; r12=0.6
n = 100000; nburn=1000; nsel = 10
#
# set characteristic MCMC step size and annealing parameter alpha
#
step = 1.0; alpha =1.0 
x = -1.0; y = -1.0
annealing = True
if annealing:
    nanneal = 10000; epsann = 2.0

chain = []
chain.append([x,y])
chann = []

# precompute a set of random numbers
nrand = n
delta = zip(rnd.uniform(-step,step,nrand),rnd.uniform(-step,step,nrand)) #random inovation, uniform proposal distribution

naccept = 0; i = 0; ntry = 0   
for id in range(n+nanneal):
    if not i%nrand:
       delta = zip(rnd.uniform(-step,step,nrand),rnd.uniform(-step,step,nrand)) #random inovation, uniform proposal distribution
       i = 0  
    xtry = x + delta[i][0] # trial step
    ytry = y + delta[i][1]
    gxtry = modelpdf(xtry,ytry,s1,s2,r12)
    gx = modelpdf(x,y,s1,s2,r12)
    if gxtry > gx: 
        x = xtry; y=ytry
        if id < n: naccept += 1
    else:     
        aprob = alpha * gxtry/gx # acceptance probability
        u = rnd.uniform(0,1)
        if u < aprob:
            x = xtry; y= ytry
            if id < n: naccept += 1
    if  id < n:
        chain.append([x,y])
        ntry +=1
    else:
        if not id%100:
            alpha = np.max(1.e-6,alpha*(1.0-1.0*(id-n+1)/nanneal)**epsann)
        chann.append([x,y])
    i += 1
    
print "Generated n ",n," samples"
print "with acceptance ratio", 1.0*naccept/ntry

if annealing:
    xann = np.array(zip(*chann)[0]); yann = np.array(zip(*chann)[1])
    xave = np.sum(xann)/nanneal; yave = np.sum(yann)/nanneal
    print "maximum from annealing: xmax, ymax=", "%6.4f"%xave, "%6.4f"%yave
#
# find maximum by annealing
#

#            
# plot results:
#
x = y = np.arange(-4.0,4.0,0.1)
X, Y = np.meshgrid(x,y)
Z = modelpdf(X,Y,s1,s2,r12)


plt.rc('font', family='sans-serif', size=16)
fig=plt.figure(figsize=(10,15))
plt.subplot(311)
plt.title('Metropolis')
plt.plot(chain)


plt.subplot(312)
xh = np.array(zip(*chain)[0]); yh = zip(*chain)[1]
#
# remove burn in samples and subselect every nsel chain entry
#
xh = xh[nburn::nsel]; yh = yh[nburn::nsel] 

plt.hist2d(xh,yh, bins=40, norm=LogNorm(), normed=1)
plt.colorbar()

dlnL2= np.array([2.30, 4.61, 9.21]) # contours enclosing 68.27, 90, and 99% of the probability density
Lmax = modelpdf(0.0,0.0,s1,s2,r12)
lvls = Lmax/np.exp(0.5*dlnL2)
cs=plt.contour(X,Y,Z, linewidths=(1.0,2.5,5.0), colors='black', norm = LogNorm(), levels = lvls, legend='target pdf' )
plt.xlim(-4,4); plt.ylim(-4,4)

plt.title('MCMC samples vs target distribution')

labels = ['68.27%', '90%','99%']
for i in range(len(labels)):
    cs.collections[i].set_label(labels[i])

plt.legend(loc='lower right')
plt.ylabel('y')
plt.xlabel('x')

plt.subplot(313)
xh = zip(*chann)[0]; yh = zip(*chann)[1]
plt.hist2d(xh,yh, bins=40, norm=LogNorm(), normed=1)
plt.colorbar()


plt.show()
