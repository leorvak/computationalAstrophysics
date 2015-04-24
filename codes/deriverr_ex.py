#
#  first derivative in different approximation and associated errors
#
# see http://matplotlib.org/users/pyplot_tutorial.html
# for matplotlib tutorial
#
#
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

def func(x):
#    return x**5
    return np.exp(x)
    
def dfuncdx(x):
#    return 5.0*x**4
    return np.exp(x)   

def der1p(x,h):
    return (func(x+h)-func(x))/h


# set array of steps
ha = np.arange(-15.0,-1.0,0.1); h=10.0**ha

# corresponding vector of x and analytic derivative to compare to
x = np.zeros(len(ha)-1); x=10.0; 
dfdx = dfuncdx(x)

# compute approximate derivative
d1= der1p(x,h)

# compute fractional error
err1 = abs((d1-dfdx)/dfdx) 

#
# now plot fractional error as a function of step
#   



fig1 = plt.figure()
plt.plot(ha,np.log10(err1),linewidth=1.5,c='r',label='1st order')

plt.xlabel('$h$')
plt.ylabel('frac. error')
plt.title('derivative error')
plt.legend(loc='lower left')
plt.savefig('deriverr.pdf')
plt.show()
