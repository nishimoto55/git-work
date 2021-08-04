# -*- coding: utf-8 -*-
"""
2021.8.2 Nishimoto  - treatment of Text Image using Numpy -
2021.8.4 Nishimoto


Start
"""
#Reading Library
import matplotlib.pyplot as plt
import numpy as np
import cv2

"""
GLOBAL VARIABLES
"""
#const.
#binarization
thre = 70 #Threshold
FontSize = 14

#Loading
datafile = './TextImage/string2'
#datafile = './TextImage/zigzag' 

data = np.loadtxt(datafile)
cv2.imwrite('./JPEG/sample.jpg',data)

"""
Function
"""
#Gray Scale
def rgb2gray(img):
    _img = img.copy().astype(np.float32)
    gray = _img[..., 0] * 0.2126 + _img[..., 1] * 0.7152 + _img[..., 2] * 0.0722
    gray = np.clip(gray, 0, 255)
    return gray.astype(np.uint8)

#Binarization
def binary(img,th):
    _img = img.copy()
    _img = np.minimum(_img // th, 1)*255
    return _img.astype(np.uint8)

#diplay histogram
def histogram(img,img2):
    plt.figure(figsize = (12,3))
    #1
    plt.subplot(1,3,1)
    plt.hist(image_origin.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.title('origin', size = FontSize)
    plt.ylim(0,10000)
    plt.xlabel('pixel',size = FontSize); plt.ylabel('appearance', size = FontSize)
    plt.title('origin', size = FontSize)
    #2
    plt.subplot(1,3,2)
    plt.title('gray scale' , size = FontSize)
    plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.ylim(0,10000)
    plt.xlabel('pixel',size = FontSize); plt.ylabel('appearance', size = FontSize)
    plt.title('normalize', size = FontSize)
    #3
    plt.subplot(1,3,3)
    plt.title('binarization' , size = FontSize)
    plt.hist(img2.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.ylim(0,10000)
    plt.xlabel('pixel',size = FontSize); plt.ylabel('appearance', size = FontSize)

    plt.tight_layout()

    plt.show()

#normalization histogram
def hist_normalize(img, a, b):
    c, d = img.min(), img.max()
    # if c <= xin < d
    out = (b - a) / (d - c) * (img - c) + a
    # if xin < c
    out[img < c] = a
    # if xin > d
    out[img > d] = b
    return np.clip(out, 0, 255).astype(np.uint8)



"""
Main
"""
#type of image_* is numpy.ndarray

image_origin = cv2.imread('./JPEG/sample.jpg')
image_gray = rgb2gray(image_origin)
image_hist_norm = hist_normalize(image_gray,a=0,b=255)

image_binary = binary(image_hist_norm, thre)

#histogram(image_hist_norm,image_binary)

"""
display
"""
#analysis
plt.figure(figsize = (12,3))
#1
plt.subplot(1,3,1)
plt.title('gray scale', size = FontSize)
plt.axis('off')
plt.imshow(image_gray,cmap = 'viridis')
plt.colorbar()
#2
plt.subplot(1,3,2)
plt.title('normalize' , size = FontSize)
plt.axis('off')
plt.imshow(image_hist_norm,cmap = 'viridis')
plt.colorbar()
#3
plt.subplot(1,3,3)
plt.title('binarization', size = FontSize)
plt.axis('off')
plt.imshow(image_binary, cmap = 'viridis')
plt.colorbar()

plt.tight_layout()

plt.show()

count_lst, hir_lst = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(count_lst))

for cnt in count_lst:
    Find_ions = cv2.drawContours(image_binary, [cnt], -1, (255,0,0), 1)
plt.imshow(Find_ions)
