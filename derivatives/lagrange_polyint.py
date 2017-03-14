#
# interpolate using Lagrangian interpolation
#

import numpy as np

def LagrangeInterp(data, x):
    #Number of data points
    n=len(data)
    #Number of x points
    nx = len(x)

    #Parse x, y data points
    dx = [d[0] for d in data]
    dy = [d[1] for d in data]

    #Allocate space for L(x)
    L = [0.0]*(nx)

    def b(j,xi):
        """Calculate b_j(x_xi)"""
        v = 1.0
        for k in xrange(n):
            if k != j:
                v *= (xi-dx[k]) / (dx[j]-dx[k])
        return v

    #Construct L(x)
    for i,xi in enumerate(x):
        #Construct each element of L(x)
        for j in xrange(n):
            L[i] += dy[j]*b(j,xi)

    return L

if '__main__' in __name__:
    import matplotlib.pylab as plt

    n=5
    LX=x(250)
    x = lambda n: np.linspace(-1,1,n)
    f = lambda x: np.cos(np.sin(np.pi*x))

    plt.plot(x(300),f(x(300)),'k')

    data=zip(x(n),f(x(n)))
    LY = LagrangeInterp(data, LX)

    plt.plot(LX,LY,'r')
    plt.show()
