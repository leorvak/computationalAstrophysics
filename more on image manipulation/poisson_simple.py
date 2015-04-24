#
#  simple Poisson eq. solver
#
import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
from matplotlib import pylab as plt

# grid size
N=64

x1d = np.linspace(0, N-1, N)
x, y, z = np.meshgrid(x1d, x1d, x1d)

xc = yc = zc = N/2
sx2 = sy2 = sz2 = 1.0
rho = 1.0*np.exp(-0.5*((x-xc)**2/sx2+(y-yc)**2/sy2+(z-zc)**2)/sz2)

rhof = fftn(rho) / N**3

G1d = fftfreq(N) * np.pi/N

kx, ky, kz = np.meshgrid(G1d, G1d, G1d)
G2 = (np.sin(kx)**2+np.sin(ky)**2+np.sin(kz)**2)
G2[0, 0, 0] = 1  # omit the G=0 term

tmp = -0.25*rhof / G2
tmp[0, 0, 0] = 0  # omit the G=0 term

phi = np.real(ifftn(tmp))

plt.figure()
plt.imshow(np.log10(np.clip(np.abs(rho[:,:,N/2]),1.e-10,1.e100)), cmap='winter' )
plt.figure()
plt.imshow(np.log10(np.clip(np.abs(phi[:,:,N/2]),1.e-10,1.e100)), cmap='autumn' )
plt.figure()
plt.plot(phi[:,N/2,N/2])
plt.show()
