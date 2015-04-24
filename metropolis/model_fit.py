import math
import numpy as np
from scipy import interpolate as intp
from scipy import integrate as integrate
from matplotlib import pylab as plt
from matplotlib.colors import LogNorm
from numpy import random as rnd
import scipy.stats as stats;
import sys

from bcgdata import read_bcg_data

def printf(format, *args):
    sys.stdout.write(format % args)


def modelpdf (md, cd, sd, x, ex, y, ey):
    """
    likelihood for a linear model for data with error bars in both directions 
    and intrinsic scatter in y direction
    
    input: xv - vector of parameters: 
           xv[0] = slope m; xv[1] = intercept c; xv[2]=intrinsic scatter
           d - data tuples of x y            
    """
    dummy = sd**2 + ey**2 + md**2*ex**2
    modelpdf = np.exp(-0.5*(np.sum(np.log(np.sqrt(dummy)))+np.sum((y-md*x-cd)**2/dummy)))
    return modelpdf

x,ex,y,ey = read_bcg_data()

ax = 14.5; x = x - ax
ay = 12.5; y = y - ay

#avrhalf=sum(rhalf/erh**2.0)/sum(1./erh**2.0)
#ay=sum(y)/len(y)

print "Pivot is at: x=",ax, "  ay=",ay

nwalkers = 1
n = 100000; nburn=0; nsel=1; step = 0.1
m0 = 1.0; c0 = 0.0; s0 = 0.1
m = rnd.normal(m0,step,nwalkers); c = rnd.normal(c0,step,nwalkers)
s = rnd.normal(s0,step,nwalkers)
chain = []

# precompute a set of random numbers
nrand = n*nwalkers
# use uniform proposal distribution
delta = zip(rnd.uniform(-step,step,nrand),rnd.uniform(-step,step,nrand),rnd.uniform(-step,step,nrand)) 

naccept = 0; i = 0; ntry = 0   
for nd in range(n):
    for i in range(nwalkers):
        mtry = m[i] + delta[ntry][0] # trial step
        ctry = c[i] + delta[ntry][1]
        stry = s[i] + delta[ntry][2]
        gxtry = modelpdf(mtry,ctry,stry,x,ex,y,ey)
        gx = modelpdf(m[i],c[i],s[i],x,ex,y,ey)
        if gxtry > gx: 
            m[i] = mtry; c[i]=ctry; s[i] = stry
            naccept += 1
        else:     
            aprob = gxtry/gx # acceptance probability
            u = rnd.uniform(0,1)
            if u < aprob:
                m[i] = mtry; c[i]= ctry; s[i] = stry
                naccept += 1
        if nd > nburn and (not nd%nsel) : # start the chain only after burn in
            chain.append([m[i],c[i],s[i]])
        ntry += 1
    
    
print "Generated n ",n*nwalkers," samples using", nwalkers," walkers"
print "with acceptance ratio", 1.0*naccept/ntry

mh = zip(*chain)[0]; ch = np.array(zip(*chain)[1]); sh = np.array(zip(*chain)[2]); sh = np.sqrt(sh*sh)

def pstats(x):
    xmed = np.median(x); xm = np.mean(x); xsd = np.std(x)
    xcfl11 = np.percentile(x,16); xcfl12 = np.percentile(x,84)
    xcfl21 = np.percentile(x,2.5); xcfl22 = np.percentile(x,97.5)

    printf('mean, median = %7.5f, %7.5f\n',xm, xmed)
    printf('st.dev = %7.5f\n',xsd)
    printf('68perc interval = %7.5f, %7.5f\n',xcfl11,xcfl12)
    printf('95perc interval = %7.5f, %7.5f\n',xcfl21,xcfl22)

print "=================="
print "best fit slope:"
pstats(mh)
print "best fit normalization (pivoted) to ",ay
pstats(ch)
print "best fit scatter in y direction:"
pstats(sh)
print "=================="

#            
# plot results:
#
    

plt.rc('font', family='sans-serif', size=16)
fig=plt.figure(figsize=(6,10))
plt.subplot(211)
plt.title('Metropolis')
plt.plot(chain)

ax = plt.subplot(212)
plt.hist2d(mh,ch, bins=60, norm=LogNorm(), normed=1)
plt.colorbar()

plt.legend(loc='upper center')
plt.ylabel('c')
plt.xlabel('m')

plt.show()
