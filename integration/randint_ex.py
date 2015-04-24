import random
from numpy import random as rnd

n = 10
#random.seed(42)
random.seed()
israndom = [0]*n
for i in range(n): israndom[i]=random.randint(1,n)

print "sequence from random:"
print israndom

inprand = [0]*n
#rnd.seed(42)
rnd.seed()
for i in range(n): inprand[i] = rnd.randint(1,n)

print "sequence from numpy.random:"
print inprand
