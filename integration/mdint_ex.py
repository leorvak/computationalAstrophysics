#
#   example of multi-dimensional integral from stackoverflow
#   by the triple-quad routine. 
#

import time
import numpy
import scipy.integrate
import math

def w(r, theta, phi, alpha, beta, gamma):
    return(-math.log(theta * beta))

def integrand(phi, alpha, gamma, r, theta, beta):
    ww = w(r, theta, phi, alpha, beta, gamma)
    k = 1.
    T = 1.
    return (math.exp(-ww/(k*T)) - 1.)*r*r*math.sin(beta)*math.sin(theta)

# limits of integration

def zero(x, y=0):
    return 0.

def one(x, y=0):
    return 1.

def pi(x, y=0):
    return math.pi

def twopi(x, y=0):
    return 2.*math.pi

# integrate over phi [0, Pi), alpha [0, 2 Pi), gamma [0, 2 Pi)
def secondIntegrals(r, theta, beta):
    res, err = scipy.integrate.tplquad(integrand, 0., 2.*math.pi, zero, twopi, zero, pi, args=(r, theta, beta))
    return res

# integrate over r [0, 1), beta [0, 2 Pi), theta [0, 2 Pi)
def integral():
    return scipy.integrate.tplquad(secondIntegrals, 0., 2.*math.pi, zero, twopi, zero, one)


t0 = time.time()

expected = 16*math.pow(math.pi,5)/3.
result, err = integral()
diff = abs(result - expected)

t1 = time.time()

print "Result = ", result, " estimated error = ", err
print "Known result = ", expected, " error = ", diff, " = ", 100.*diff/expected, "%"
print "completed in time ",t1-t0," sec"