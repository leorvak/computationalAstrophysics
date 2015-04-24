#
# interpolate using Lagrangian interpolation
#

import numpy as np
from scipy import interpolate as int


if '__main__' in __name__:
    import matplotlib as mpl
    import matplotlib.pylab as plt
    
    plt.switch_backend('TkAgg')
    plt.ion()
    x = lambda n: np.linspace(-1,1,n)
    f = lambda x: np.cos(np.sin(np.pi*x))
    xd = x(300); fd=f(xd)
    plt.plot(xd,fd,'k')

    spf = int.interp1d(x(300),fd,kind='cubic')
    
    plt.plot(xd,spf(xd),'r')
