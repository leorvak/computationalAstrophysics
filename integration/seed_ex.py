#
#  example illustrating how to initialize different random sequences. and that these
#  need to be seeding with different seeds.
#
import random
import time

print 'Default initialization:\n'

r1 = random.Random()
r2 = random.Random()

n = 5
for i in xrange(n):
    print '%04.3f  %04.3f' % (r1.random(), r2.random())

print '\nSame seed:\n'

seed = time.time()
r1 = random.Random(seed)
r2 = random.Random(seed)

for i in xrange(n):
    print '%04.3f  %04.3f' % (r1.random(), r2.random())


#
# use jumpahead to force second sequence into a different part of RNG period
#
print '\nUsing jumpahead:\n'

r1 = random.Random()
r2 = random.Random()

# Force r2 to a different part of the random period than r1.
r2.setstate(r1.getstate())
r2.jumpahead(1024)

for i in xrange(n):
    print '%04.3f  %04.3f' % (r1.random(), r2.random())
    
print '\nSystemRandom: Default initialization:\n'

r1 = random.SystemRandom()
r2 = random.SystemRandom()

for i in xrange(n):
    print '%04.3f  %04.3f' % (r1.random(), r2.random())

print '\nSystemRandom: Same seed:\n'

seed = time.time()
r1 = random.SystemRandom(seed)
r2 = random.SystemRandom(seed)

for i in xrange(n):
    print '%04.3f  %04.3f' % (r1.random(), r2.random())
    
nlarge = 10000000
t0 = time.time()
for i in range(nlarge): r1.random()
t1 = time.time()

print "it took SystemRandom", t1-t0,"sec to do",nlarge," random numbers"

r2 = random.Random()

t0 = time.time()
for i in range(nlarge): r2.random()
t1 = time.time()

print "it took random.Random", t1-t0,"sec to do",nlarge," random numbers"
