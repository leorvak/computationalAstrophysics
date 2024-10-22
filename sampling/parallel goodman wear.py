#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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
from time import time
from time import sleep
from mpi4py import MPI

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

#if MPI is None:
#    raise ImportError("Failed to import MPI from mpi4py")

#  start MPI
#
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

fname = "proc"+str(rank)+".out"
fl = open(fname,"w")
fl.write("process started work...")
fl.close()

nparams = 2
n = 100; nburn=0; nsel=1; step = 0.1
ap = 2.0; api = 1.0/ap; asqri=1.0/np.sqrt(ap); afact=(ap-1.0)
nwalkers = 100

#if ( nwalkers%2 not 0 ):
#    raise RuntimeError("nwalkers must be divisible by 2")

#if ( (nwalkers/2)%size not 0 ):
#    raise RuntimeError("nwalkers/2 must be exactly divisible by nproc")

nchunk = nwalkers/2/size

if ( rank == 0 ):
    fl = open(fname,"a")
    print>>fl, "nchunk =", nchunk
    fl.close()

if ( rank == 0 ):
    rnd.seed(651)


fl = open(fname,"a")
print >>fl, "process", rank," created random seed"
fl.close()

#
# distribute initial positions of walkers in an isotropic Gaussian around the initial point
#

if ( rank == 0 ): 
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
        if ( rank == 0 ):
            xcompl = x[kd,:,:]
            for nproc in range(1,size):
                i1 = rank*nchunk; i2 = (rank+1)*nchunk
                xchunk = x[k,i1:i2,:]
                comm.send(xchunk,dest=nproc, tag=12345)
                comm.send(xcompl,dest=nproc,tag=54321)
            #its own chunk and complement
            xchunk = x[k,0:nchunk,:]
        else:
            xchunk = comm.recv(source=0,tag=12345)
            xcompl = comm.recv(source=0,tag=54321)
        # execute the loop
        naccloc = 0 
        for i in range(nchunk):
            zf= rnd.rand()   # the next few steps implement Goodman & Weare sampling algorithm
            zr = (1.0+zf*afact)**2*api
            j = rnd.randint(nwalkers/2)
            xtry = xcompl[j,:] + zr*(xchunk[i,:]-xcompl[j,:])
            gxtry = modelpdf(xtry)
            gx = modelpdf(xchunk[i,:])
            gx = np.max(np.abs(gx),1.e-50)
            aprob = zr*gxtry/gx
            if aprob >= 1.0:
                xchunk[i,:] = xtry
                naccloc += 1
            else:
                u = rnd.uniform(0,1)
                if u < aprob:
                    xchunk[i,:] = xtry
                    naccloc += 1
        # now collect chunks
        if ( rank == 0 ):
            x[k,0:nchunk,:] = xchunk
            naccept += naccloc
            for nproc in range(1,size):
                xchunk = comm.recv(source=nproc,tag=12345)
                nal = comm.recv(source=nproc,tag=123)
                i1 = rank*nchunk; i2 = (rank+1)*nchunk
                x[k,i1:i2,:]=xchunk
                naccept += nal
        else:
            comm.send(xchunk,dest=0,tag=12345)
            comm.send(naccloc,dest=0,tag=123)
        # at this point x array should be whole
        # let's do some statistics with it
        if rank == 0:
            if ( nchain >= nburn ):
                for i in range(nwalkers/2):
                    if ( nchain == 199 ):
                        print x[k,i,:]
                    chain.append(np.array(x[k,i,:]))
                    mw[k*nwalkers/2+i,:] += x[k,i,:]
                    sw[k*nwalkers/2+i,:] += x[k,i,:]**2
                    ntry += 1

    if  rank == 0:
        nchain += 1
        if nchain >= nburn and nchain > 2 and nchain%nRval == 0:
            # use Gelman & Rubin convergence instead of the flaky corr. time
            nchainb = nchain - nburn
            mwc = mw/(nchainb-1.0)
            swc = sw/(nchainb-1.0)-np.power(mwc,2)

            for i in range(nparams):
                # within chain variance
                Wgr[i] = np.sum(swc[:,i])/nwalkers
                # mean of the means over Nwalkers
                m[i] = np.sum(mwc[:,i])/nwalkers
                # between chain variance
                Bgr[i] = nchainb*np.sum(np.power(mwc[:,i]-m[i],2))/(nwalkers-1.0)
                # Gelman-Rubin R factor
                Rgr[i] = (1.0 - 1.0/nchainb + Bgr[i]/Wgr[i]/nchainb)*(nwalkers+1.0)/nwalkers - (nchainb-1.0)/(nchainb*nwalkers)
            Rval.append(Rgr-1.0)
            fl = open(fname,"a")
            print >>fl, "nchain=",nchain
            print >>fl, "R values for parameters:", Rgr
            print >>fl, mwc, m
            fl.close()
            if np.max(Rgr-1.0) < 0.05: converged = True

    if rank == 0:
        for nproc in range(1,size):
            comm.send(converged,dest=nproc,tag=777)
    else:
        converged = comm.recv(source=0,tag=777)
        if converged:
            fl = open(fname,"a")
            print >>fl, " proc",rank,"is told that chain converged"
            print >>fl, "finishing..."
            fl.close()
    if ( rank == 0 and converged ):
        fl = open(fname,"a")
        print >>fl, "proc 0 finalizing:"
        print >>fl, "Generated ",ntry," samples using", nwalkers," walkers"
        print >>fl, "with step acceptance ratio of", 1.0*naccept/ntry
        fl.close()
        xh = zip(*chain)[0]; yh=zip(*chain)[1]
        # write the chain into file
        fc = file("chain.dat","w")
        for i, xd in enumerate(xh):
            print >>fc, xd, yh[i]
        fc.close()

