#!/usr/bin/env python

#
# Use linalg.lstsq to fit a line

import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt
import socket

from bcgdata import read_bcg_data

x,ex,y,ey = read_bcg_data()

#
# pivot
#
ax = 14.5; x = x - ax
ay = 12.5; y = y - ay

A = np.array([x,np.ones(len(x))])
w = np.linalg.lstsq(A.T,y)[0]
A2 = np.array([np.exp(x),x**2,x,np.ones(len(x))])
c = la.lstsq(A2.T,y)[0]
#c,resid,rank,sigma = linalg.lstsq(A,y)

xf = np.linspace(-1,1,100)
yf  = w[0]*xf + w[1]
yf2 = c[2]*xf + c[3]
yf3 = c[0]*np.exp(xf) + c[1]*xf**2 + c[2]*xf + c[3]
print "numpy slope=",w[0]," intercept=",w[1]
print "scipy slope=",c[2]," intercept=",c[3]
print "cube, square coeff=", c[0], c[1]
#            
# plot results:
#

plt.plot(x,y,'ro',xf,yf)
plt.plot(xf,yf3)
plt.plot(xf,yf3)
plt.xlabel(r'$M_{500} (M_{\odot})$')
plt.ylabel(r'$M_{*,tot} (M_{\odot})$')
plt.title('Data fitting with linalg.lstsq')
plt.show()
