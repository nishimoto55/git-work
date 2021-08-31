# -*- coding: utf-8 -*-
"""
2021.8.2 Nishimoto - treatment of Text Image using Numpy -
2021.8.4 Nishimoto
2021.8.5 Nishimoto - Detection ions -
"""
# Module
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
"""
GLOBAL VARIABLES
"""
#const.
circle_thre = 30 #threshold for determining number of ions
thre = 120  #Threshold for binarization
FontSize = 14 #For graph
l=4 #File index (e.g.) -> files[l]

#Loading Raw-File
files = glob.glob('./TextImage/s*')

print('files is ...')
for file in files:
    print(file)

#textimage2jpg
data = np.loadtxt(files[l])
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


#diplay histogram
def histogram(img1,img2,img3):
    ymax = 300
    plt.figure(figsize = (12,3))
    #1
    plt.subplot(1,3,1)
    plt.hist(img1.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.title('origin', size = FontSize)
    plt.ylim(0,ymax)
    plt.xlabel('pixel',size = FontSize); plt.ylabel('appearance', size = FontSize)
    plt.title('gray scale', size = FontSize)
    #2
    plt.subplot(1,3,2)
    plt.title('gray scale' , size = FontSize)
    plt.hist(img2.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.ylim(0,ymax)
    plt.xlabel('pixel',size = FontSize); plt.ylabel('appearance', size = FontSize)
    plt.title('normalize', size = FontSize)
    #3
    plt.subplot(1,3,3)
    plt.title('binarization' , size = FontSize)
    plt.hist(img3.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.ylim(0,ymax)
    plt.xlabel('pixel',size = FontSize); plt.ylabel('appearance', size = FontSize)
    plt.tight_layout()
    plt.savefig('./JPEG/COMPARE_histogram.jpg',
            bbox_inches='tight',
            dpi = 300)
    plt.show()
    
def origin():
    plt.figure(figsize = (12,3))
    #image
    ymax = 300
    plt.subplot(1,2,1)
    plt.title('origin', size = FontSize)
    plt.axis('off')
    plt.imshow(image_origin,cmap = 'gray'); plt.colorbar()
    #histogram
    plt.subplot(1,2,2)
    plt.hist(image_origin.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.title('origin', size = FontSize)
    plt.ylim(0,ymax)
    plt.xlabel('pixel',size = FontSize); plt.ylabel('appearance', size = FontSize)
    plt.title('origin histogram', size = FontSize)
    plt.tight_layout()
    plt.savefig('./JPEG/RAW-image.jpg',
            bbox_inches='tight',
            dpi = 300)
    plt.show()
    

def display(img1,img2,img3):
    plt.figure(figsize = (12,3))
    i = 3; j=0 
    #1
    j+=1
    plt.subplot(1,i,j)
    plt.title('gray scale', size = FontSize)
    plt.axis('off')
    plt.imshow(img1,cmap = 'viridis'); plt.colorbar()
    #2
    j+=1
    plt.subplot(1,i,j)
    plt.title('normalize' , size = FontSize)
    plt.axis('off')
    plt.imshow(img2,cmap = 'viridis'); plt.colorbar()
    #3
    j+=1
    plt.subplot(1,i,j)
    plt.title('binarization', size = FontSize)
    plt.axis('off')
    plt.imshow(img3, cmap = 'viridis'); plt.colorbar()
    plt.savefig('./JPEG/gasyo_image.jpg',
            bbox_inches='tight',
            dpi = 300)
    plt.tight_layout(); plt.show()
    
"""
start Main
"""
#Load raw-image and gray scale, normalization, binarization
image_origin = cv2.imread('./JPEG/sample.jpg')
image_gray = rgb2gray(image_origin)
image_hist_norm = hist_normalize(image_gray,a=0,b=255)
image_binary = binary(image_hist_norm, thre)

#origin()

### show HISTOGRAM
#histogram(image_gray,image_hist_norm,image_binary)

### show image
#display(image_gray,image_hist_norm,image_binary)


#イオンの検出
count_lst, hir_lst = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

list = np.array([[0]*2]*6)

print('position of ions is ...')
img = image_origin
k=0 #count variable
for i, cnt in enumerate(count_lst):
        # 輪郭の面積を計算する。
        area = cv2.contourArea(cnt)
        #　抽出する範囲を指定
        if area > circle_thre and area < 10000:
            #最小外接円を計算する
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            center = (int(x),int(y))
            #輪郭の円を描画
            cv2.circle(img,center,int(radius),(0,0,255),5)
            #中心点に円を描画
            list[k][0] = int(x)
            list[k][1] = int(y)
            cv2.circle(img,center,1,(0,255,0),1)
            print(str(k+1) + 'th (x,y) : ' ,'{:.3f}'.format(x),'{:.3f}'.format(y))
            print('circle area is ' + str(area))
            k=k+1

for i in list:
    print(i)

cv2.imwrite('./JPEG/out.jpg',img)
   
"""
end Main
"""