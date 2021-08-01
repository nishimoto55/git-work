from PIL import Image, ImageFilter
from matplotlib import pyplot

im = Image.open('sample.jpg')

img_mod = where(im<120,0,im)

pyplot.imshow(img_mod)
