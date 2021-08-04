# -*- coding: utf-8 -*-
"""
2021.8.3 Nishimoto - getting histogram with OpenCV

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 #OpenCV

file = './sample.jpg'


def show_img(path):

    img = cv2.imread(path)
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    hist_b = cv2.calcHist([b],[0],None,[256],[0,256])
    hist_g = cv2.calcHist([g],[0],None,[256],[0,256])
    hist_r = cv2.calcHist([r],[0],None,[256],[0,256])
    plt.plot(hist_r, color='r', label="r")
    plt.plot(hist_g, color='g', label="g")
    plt.plot(hist_b, color='b', label="b")
    plt.legend()
    plt.show() 
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = img2[:,:,0], img2[:,:,1], img2[:,:,2]
    hist_h = cv2.calcHist([h],[0],None,[256],[0,256])
    hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
    hist_v = cv2.calcHist([v],[0],None,[256],[0,256])
    plt.plot(hist_h, color='r', label="h")
    plt.plot(hist_s, color='g', label="s")
    plt.plot(hist_v, color='b', label="v")
    plt.legend()
    plt.show()

    return hist_r,hist_g, hist_b, hist_h, hist_s, hist_v



r,g,b,h,s,v = show_img("sample.jpg")

"""

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


"""
datafile = './3ions'

image = np.loadtxt(datafile)
image_list = np.array(image)

#type is <class 'numpy.ndarray'>
image_hist,image_bins = np.histogram(image_list.flatten(),bins=np.arange(256+1))

plt.plot(image_hist)
plt.show()

"""