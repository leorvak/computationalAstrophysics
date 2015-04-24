#
#  example derived from one of the SciPy examples
#  at http://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/fftpack.html
#
#  illustrates that DFT is essentially an interpolation of a function
#

import numpy as np
from numpy.fft import fft, ifft
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt

def func2(x):
    return np.exp(-x/3.0)*np.cos(2.0*x)

def func(x):
    fdum = np.zeros_like(x)
    for i, xd in enumerate(x):
        if (xd < 10. and xd >= 0.):
            fdum[i] = 1.0
    return fdum
    
N = 100
x = np.linspace(0,20,N)

fx = func(x)

klim = 10
fk = dct(fx, norm='ortho')
fr = idct(fk, norm='ortho')
window=np.zeros(N); window[:klim]=1.0
frlim = idct(fk*window, norm='ortho')
print sum(abs(fx-fr)**2) / sum(abs(fx)**2)


plt.plot(x, fx, '-b')
plt.plot(x, frlim,'g')
plt.plot(x, fr, 'r')

#plt.plot(t, yr, 'g+')
#plt.legend(['x', '$x_{20}$', '$x_{15}$'])
plt.grid()
plt.show()