#
#  illustration of the concept of burn-in arising when the initial sample is chosen in the region of low probability
#  for discussion see: http://users.stat.umn.edu/~geyer/mcmc/burn.html
#
from numpy import random as rnd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn


# number of samples
n = 1000

#initial conditions and parameters
x10 = 10.0; x20 = 0
r=0.98; s = 0.2

# generate a string of Gaussian numbers
en = rnd.normal(0.0,s,n)

x1 = np.copy(en); x2 = np.copy(en)

# set the starting point
x1[0] += x10; x2[0] += x20

# iterate
for i in range(1,n):
    x1[i] += r*x1[i-1]; x2[i] +=r*x2[i-1]

# plot
fig = plt.figure()

plt.plot(x1,linewidth=1.5,c='r',label='x0=10.0')
plt.plot(x2,linewidth=1.5,c='b',label='x0=0.0')

plt.xlabel('iteration')
plt.ylabel('x')
plt.title('burn-in illustration')
plt.legend(loc='upper right')
#plt.savefig('burnin_ex.pdf')
plt.show()

