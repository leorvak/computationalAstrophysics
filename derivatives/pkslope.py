#
#  basic example how to read P(k) output by camb
#  and compute its slope (log derivatives) using finite differences of different order
#
# see also http://matplotlib.org/users/pyplot_tutorial.html
# for matplotlib tutorial
#
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


plt.switch_backend('TkAgg')

#
# functions implementing derivative approximations of different order 
# for arrays of x and f. Note the use of numpy vector operations on arrays 
# to compute finite differences
#
def der1p(x,f):
    der1p = (np.roll(f,-1)-f)/(np.roll(x,-1)-x)
    der1p[len(der1p)-1] = (f[len(f)-1]-f[len(f)-2])/(x[len(x)-1]-x[len(x)-2])
    return der1p

def der2(x,f):
    der2 = (np.roll(f,-1)-np.roll(f,1))/(np.roll(x,-1)-np.roll(x,1))
    der2[0]=(f[1]-f[0])/(x[1]-x[0])
    der2[len(der2)-1]=(f[len(f)-1]-f[len(f)-2])/(x[len(x)-1]-x[len(x)-2])
    return der2
    
def der4(x,f):
    der4 = ((8.0*(np.roll(f,-1)-np.roll(f,1))+np.roll(f,2)-np.roll(f,-2))/
            (6.0*(np.roll(x,-1)-np.roll(x,1))))
    der4[0] = (f[1]-f[0])/(x[1]-x[0]) # 1st order on the boundary
    der4[1] = (f[2]-f[0])/(x[2]-x[0]) # "2nd order" next to the boundary
    der4[len(x)-1]=(f[len(f)-1]-f[len(f)-2])/(x[len(x)-1]-x[len(x)-2])
    der4[len(x)-2]=(f[len(f)-1]-f[len(f)-3])/(x[len(x)-1]-x[len(x)-3])
    return der4

#
# read input file with P(k) output from CAMB
#
fname = './test_matterpower_orig.dat'
k, Pk = np.loadtxt(fname,usecols=(0,1),unpack=True)

lk = np.log10(k); lPk = np.log10(Pk)

#
# compute first derivative approximations of different order
#
dlPdlk1 = der1p(lk,lPk)
dlPdlk2 = der2(lk,lPk)
dlPdlk4 = der4(lk,lPk)

#
# plot P(k)
#
mpl.use("TKAgg")  # allow X windows on midway

plabel = '$\\log_{10} P(k)$'
fig1 = plt.figure()
#plt.ylim((2,3))
#plt.xlim((-1,-0.99))
plt.plot(lk,lPk,linewidth=1.5,c='b',label=plabel)

plt.xlabel('$\\log_{10} k$')
plt.ylabel('$\\log_{10} P(k)$')
plt.title('power spectrum')
plt.legend()

#
# plot derivative approximations
#
fig2 = plt.figure()
plt.plot(lk,dlPdlk1,linewidth=1.5,c='r',label='1st order')
plt.plot(lk,dlPdlk2,linewidth=1.5,c='b',label='2nd order')
plt.plot(lk,dlPdlk4,linewidth=1.5,c='g',label='4th order')

plt.xlabel('$\\log_{10} k$')
plt.ylabel('$d\\ln P(k)/d\\ln k$')
plt.title('slope')
plt.legend()

plt.show()
