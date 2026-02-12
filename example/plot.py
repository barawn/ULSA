#! /usr/bin/env python3

import numpy as np
from matplotlib import pylab as plt
import healpy as hp
import h5py
import sys


f = sys.argv[1]

dataset = h5py.File(f)
data = np.array(dataset['data'])
hp.mollview(data,norm='log',title=f)
plt.show()

