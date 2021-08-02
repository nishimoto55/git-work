# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 18:21:57 2021

@author: 井上 凌太
"""

import numpy as np
import matplotlib.pyplot as plt

datafile = './3ions'

image = np.loadtxt(datafile)
image_list = np.array(image)

#type is <class 'numpy.ndarray'>
image_hist,image_bins = np.histogram(image_list.flatten(),bins=np.arange(256+1))

plt.plot(image_hist)
plt.show()
