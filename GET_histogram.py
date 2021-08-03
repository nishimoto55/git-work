# -*- coding: utf-8 -*-
"""
2021.8.3 Nishimoto - getting histogram with OpenCV

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 #OpenCV

file = './sample.jpg'


#画像をグレースケールで読み込む
img = cv2.imread(file,0)

#一次元ヒストグラムを生成する
n_bins = 100
hist_range = [0,256]

hist = cv2.calcHist([img],
                    channels = [0],
                    mask = None,
                    histSize = [n_bins],
                    ranges = hist_range)
hist = hist.squeeze(axis=-1)

def plot_hist(bins,hist,color):
    centers = (bins[:-1] + bins[1:])/2
    widths = np.diff(bins)
    ax.bar(centers,hist, width=widths, color=color)
    
bins = np.linspace(*hist_range, n_bins + 1)

fig, ax = plt.subplots()
ax.set_xticks([0,256])
ax.set_yticks([0,256])
plot_hist(bins,hist,color='k')
plt.show()




"""
datafile = './3ions'

image = np.loadtxt(datafile)
image_list = np.array(image)

#type is <class 'numpy.ndarray'>
image_hist,image_bins = np.histogram(image_list.flatten(),bins=np.arange(256+1))

plt.plot(image_hist)
plt.show()

"""