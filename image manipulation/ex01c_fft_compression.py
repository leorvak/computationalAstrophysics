#!/usr/bin/env python

# Author: Andrey Kravtsov; inspired by the SVD compression code by Douglas Rudd
# Date: 10/27/2014
#
# Apply FFT to crudely compress a greyscale image
# image. Requires numpy and PIL (or Pillow)

import numpy as np
from scipy.fftpack import ifftn
from scipy.fftpack import fftn
from PIL import Image

# load example image as numpy array
img = np.asarray(Image.open("spt.png"))

# compute FFT of image
zfft = fftn(img)

# Generate images with differing compression levels
for k in 10, 50, 100, 200:
    # reconstruct image using k components of decomposition
    zfftd = np.copy(zfft)
    zfftd[k:,k:] = 0.0
    img_ = ifftn(zfftd)

    # compute compression ratio
    ratio = float(k*(1+sum(img.shape)))/np.prod(img.shape)
    print "Compression factor %.5f" % (ratio, )

    # save resulting image
    Image.fromarray(img_.astype(np.uint8)).save("spt_k%03d.png" % (k, ))
    #Image.fromarray(img_.astype(np.uint8)).show()