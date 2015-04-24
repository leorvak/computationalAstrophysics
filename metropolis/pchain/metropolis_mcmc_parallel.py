#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  example of the Metropolis algorithm single walker chain
#  a difficult distribution: the Rosenbrock “banana” pdf
#

import math
import numpy as np
from numpy import random as rnd
from mpi4py import MPI


def kPk_dft(y):
    """
    compute chain power spectrum (see Dunkley et al. 2005)
    via direct Fourier transform
    """
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
    """
    defines model pdf corresponding to the Rosenbrock "curved banana" density
    """
    return np.exp(-(100.*(x2-x1**2)**2+(1.0-x1)**2)/20.0)

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

print "process", rank, "started work..."

# read random seed and use it to set RNG initial state
rs = open('ranseeds.dat',"r")
seeds=rs.readlines()
rs.close()
if ( rank > len(seeds)-1 ):
    rseed = int(seeds[-1])
    print "WARNING!!! rank", rank, "larger than the number of seeds",len(seeds)-1
else:
    rseed = int(seeds[rank])

print "process", rank, "random seed:",rseed
rnd.seed(rseed)

# alternative is simply to initialize with seed equal to rank, or some multiple thereof
# rnd.seed(rank+1)
 
#
# all subsequent calculations are local to this process
#

# define chain parameters: N of chain entries, 
# nburn = N of first entries to discard, step = step size
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
    # if not accepted walker coordinate remains the same 
    # and is also added to the chain
    #
    if gxtry > gx: # always take step upwards 
        x = xtry; y=ytry
        naccept += 1
    else:     
        aprob = gxtry/gx # acceptance probability for downward step
        u = rnd.uniform(0,1)
        if u < aprob:
            x = xtry; y= ytry
            naccept += 1
    #
    # whatever the outcome, add current walker coordinates to the chain
    #
    chain.append([x,y])
    i += 1; ntry += 1

print rank, "------------------------------------------------", rank
print "Process", rank
print "Generated n ",n," samples with a single walker"
print "with acceptance ratio", 1.0*naccept/ntry

#
#  analyze and communicate the chain
#
xh = np.array(zip(*chain)[0]); yh = np.array(zip(*chain)[1])
xh = xh[nburn::nsel]; yh=yh[nburn::nsel]

print "process", rank, "computing DFT..."
#kj, Pkj = kPk_dft(xh)
print "process", rank, "done with DFT."

# now it is time to gather stones...

if rank == 0:
    master_chain=np.column_stack((xh,yh))
    # receive data from all other processes on master
    for n in range(1,size):
        data = comm.recv(source=n, tag=12345)
        xho = data[:][0]; yho = data[:][1]
        master_chain=np.vstack((master_chain,data))
        print "chain received from process", n
        f = open('t0.dat',"w")
        for it, dummy in enumerate(master_chain):
            print>>f,dummy[0], dummy[1]
        f.close()
else:
    # package and send data to master process
    data = np.column_stack((xh,yh))
    comm.send(data,dest=0, tag=12345)
    print "process", n, "chain sent to master:"
    f = open('t0%d.dat'%rank,"w")
    for it, dummy in enumerate(data):
        print>>f,dummy[0],dummy[1]
    f.close()
    
    
