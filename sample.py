from PIL import Image
from matplotlib import pyplot
import numpy as np

thre = 50

im = Image.open('sample.jpg')
im_list  = np.asarray(im)

im_list_mod = np.where(im_list<thre, 0, 255)

pyplot.imshow(im_list_mod)
