#
#    pre-generate a list of truely random numbers 
#    to be used as seeds for parallel walkers
#
#
from os import urandom as _urandom
import struct
import os
import time
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "  script requires input of number of random seeds to generate"
        sys.exit(1)

    print sys.argv[1]
    nrandom = int(sys.argv[1])
    #nrandom = 100

    # generate input number of random integers and write them into seed file
    aran = [0]*nrandom
    rs = file("ranseeds.dat","w")
    for i in range(nrandom):
        try:     
            a = struct.unpack("I",_urandom(4)) # get 4 bytes of randomness
        except:
            raise NotImplementedError('system random numbers are not available')
        
        a2 = int(time.time() * 256) # fractional seconds DO NOT USE for parallel applications
        aran[i] =a[0]
        print a[0], a2
        print >>rs, a[0]
    
    rs.close()



