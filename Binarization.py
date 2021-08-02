# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

2021.8.2 Nishimoto  - treatment of Text Image using Numpy -
"""

"""

Start

"""
#Reading Library
#from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2


datafile = './3ions' #Loading Text Image


"""
GLOBAL VARIABLES
"""
#binarization
thre = 30 #Threshold
binary_min = 0 #minimum
binary_max = 255 #maximum
#visualize binarization using matplotlib 
y_min = 600 #Default 1024
y_max = 500 #Default 0
FontSize = 18


"""
LOADING Text Image and Binarization
"""
#type is <class 'numpy.ndarray' > data and data_binary

data = np.loadtxt(datafile) 
data_binary = np.where(data<thre, binary_min, binary_max)



"""
DISPLAY Text Image(Raw and Bi) using matplotlib.pyplot
"""
plt.rcParams['font.family'] = 'Times New Roman'
#1
plt.subplot(2,1,1)
plt.ylim(y_min,y_max)
plt.axis('off')
plt.title('Raw Text Image',size = FontSize)
plt.imshow(data)
#2
plt.subplot(2,1,2)
plt.ylim(y_min,y_max)
plt.axis('off')
plt.title('Binarization', size = FontSize)
plt.imshow(data_binary)

plt.tight_layout()



"""
treatment of jpeg
"""
#im = Image.open('sample.jpg') #'type is <class 'PIL.JpegImagePlugin.JpegImageFile'>'
#im_list  = np.asarray(im) #type is <class 'numpy.ndarray' >
#im_list_mod = np.where(im_list<thre, 0, 255)
#pyplot.imshow(im_list)


