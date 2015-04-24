import time
import mcint
import random
import math

def w(r, theta, phi, alpha, beta, gamma):
    return(-math.log(theta * beta))

def integrand(x):
    r     = x[0]
    theta = x[1]
    alpha = x[2]
    beta  = x[3]
    gamma = x[4]
    phi   = x[5]

    k = 1.
    T = 1.
    ww = w(r, theta, phi, alpha, beta, gamma)
    return (math.exp(-ww/(k*T)) - 1.)*r*r*math.sin(beta)*math.sin(theta)

def sampler():
    while True:
        r     = random.uniform(0.,1.)
        theta = random.uniform(0.,2.*math.pi)
        alpha = random.uniform(0.,2.*math.pi)
        beta  = random.uniform(0.,2.*math.pi)
        gamma = random.uniform(0.,2.*math.pi)
        phi   = random.uniform(0.,math.pi)
        yield (r, theta, alpha, beta, gamma, phi)


domainsize = math.pow(2*math.pi,4)*math.pi*1
expected = 16.0*math.pow(math.pi,5)/3.

for nmc in [1000, 10000, 100000, 1000000, 10000000, 100000000]:
    t0 = time.time()
    random.seed(1)
    #random.seed()
    result, error = mcint.integrate(integrand, sampler(), measure=domainsize, n=nmc)
    diff = abs(result - expected)
    t1 = time.time()

    print "Using n = ", nmc
    print "Result = ", result, "estimated error = ", error
    print "Known result = ", expected, " error = ", diff, " = ", 100.*diff/expected, "%"
    print " "
    print "completed in time ",t1-t0,"sec"