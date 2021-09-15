# -*- coding: utf-8 -*-

# Import Module
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2

"""
GLOBAL VARIABLES
"""
#definite const.
circle_thre = 40 #threshold for determining number of ions
thre = 120  #Threshold for binarization
FontSize = 14 #For graph
l=2 #File index (e.g.) -> files[l]
#setting ROI
xmin,xmax = 0,1280
ymin,ymax = 475 ,625
#Physics Const.
e = 1.60217662 * 10 ** (-19)
A = 2.30707757*10**(-28)


#Loading Raw-Files
files = glob.glob('./TextImage/s*')

#Print List of Files
print('files is ...')
for file in files:
    print(file)

#textimage 2 jpg (one file only)
data = np.loadtxt(files[l])
cv2.imwrite('./origin_img.jpg',data)

#basis img
image_origin = cv2.imread('./origin_img.jpg')


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
    
    out = (b - a) / (d - c) * (img - c) + a
    # if xin < c
    out[img < c] = a
    # if xin > d
    out[img > d] = b
    return np.clip(out, 0, 255).astype(np.uint8)

#イオン位置における電場の導出とその図示
def ElectroField(ion_pos,k,img): #img = img1
    #配列の準備 
    sort_pos = np.array([[0.]*3]*k)
    sorted_pos = np.array([[0.]*3]*k)
    displace_x = np.array([[0.]*k]*k)
    displace_y = np.array([[0.]*k]*k)
    displace_r = np.array([[0.]*k]*k)
    
    Each_coulomb_x = np.array([[0.]*k]*k)
    Each_coulomb_y = np.array([[0.]*k]*k)
    Coulomb_x = np.array([0.]*k)
    Coulomb_y = np.array([0.]*k)
    E_x = np.array([0.]*k)
    E_y = np.array([0.]*k)
    E = np.array([0.]*k)
    
    sort_pos = ion_pos[ion_pos[:,0].argsort(), :]
    sorted_pos = sort_pos[-k:,0:2]
    
    #結果発表
    print('number of ions is ' ,k)
    print("[x,y,area]")
    for i in sorted_pos:
        print(i)
        
    np.set_printoptions(precision=4, floatmode='maxprec')

    #各イオンにおける他イオンとの距離(µm)と各軸におけるCooulomb力
    for i in range(0,k):
        for j in range(0,k):
            if i != j :         
                x = (sorted_pos[i][0] - sorted_pos[j][0])*5.3/12.9
                y = (sorted_pos[i][1] - sorted_pos[j][1])*5.3/12.9
                displace_x[i][j] = x
                displace_y[i][j] = y
                displace_r[i][j] = np.sqrt(x*x + y*y)
                #Each Coulomb force [N]
                Each_coulomb_x[i][j] = A*(displace_x[i][j])/(displace_r[i][j])**3 * 10 **(12)
                Each_coulomb_y[i][j] = A*(displace_y[i][j])/(displace_r[i][j])**3 * 10 **(12)
    
    for i in range(0,k):
        for j in range(0,k):
            Coulomb_x[i] = Coulomb_x[i] + Each_coulomb_x[i][j]
            Coulomb_y[i] = Coulomb_y[i] + Each_coulomb_y[i][j]
    
    E_x = -Coulomb_x/e
    E_y = -Coulomb_y/e
    E = np.sqrt(E_x**2 + E_y**2)
    
    
    for i in range(0,k):
        cv2.arrowedLine(img,(int(sorted_pos[i][0]),int(sorted_pos[i][1])),(int(sorted_pos[i][0]+E_x[i]*2),int(sorted_pos[i][1]+E_y[i]*2)),(255,255,0),thickness = 3,tipLength = 0.3)
       

    #結果出力
    #クーロン相互作用(1対1)
    print('distance between ions [µm]')
    print(displace_r)
    print('Coulomb force x [N]')
    print(Each_coulomb_x)
    print('Coulomb force y [N]')
    print(Each_coulomb_y)
    
    #クーロン相互作用(1対その他)=注目するイオンが他のイオンから受けるクーロン力のx,y軸それぞれについての合計
    print('sum coulomb force x[N]')
    print(Coulomb_x)
    print('sum coulomb force y[N]')
    print(Coulomb_y)
    
    #平衡の式よりイオン捕獲位置におけるx,y軸に関する電場
    print('E x[V/m]')
    print(E_x)
    print('E y[V/m]')
    print(E_y)
    #イオン捕獲位置にかかる電場の大きさ．
    print('E [V/m]')
    print(E)
    
    dst_img = image_origin.copy()
    dst_img[ymin:ymax,xmin:xmax] = img
    
    cv2.imwrite('./out_img.jpg',img)
    cv2.imwrite('./dst_img.jpg',dst_img)


#イオンの検出
def CountIons(img): #img = img1
    count_lst, hir_lst = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #各イオンのデータを格納する配列の準備  
    ion_pos = np.array([[0.]*3]*10)
    
    k=0 #count variable : イオンの個数
    
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
                cv2.circle(img,center,1,(0,255,0),1)
                #配列にイオンのx,y座標とそれぞれの面積を代入していく
                ion_pos[k][0] = int(x)
                ion_pos[k][1] = int(y)
                ion_pos[k][2] = area            
                k=k+1
    
    print(ion_pos)
    ElectroField(ion_pos,k,img)
    
    
"""
Main() {
"""
#画像処理の範囲をイオン捕獲位置に絞る
img1 = image_origin[ymin:ymax,xmin:xmax]

#切り出す範囲の明示
rect_img = image_origin.copy()
cv2.rectangle(rect_img,(xmax,ymax),(xmin,ymin),(0,255,0),2)
cv2.imwrite('rect_img.jpg',rect_img)

#切り出した範囲(img1)に対してそれぞれ処理を行う
#グレースケール化，ヒストグラムの正規化，二値化
image_gray = rgb2gray(img1)
image_hist_norm = hist_normalize(image_gray,a=0,b=255)
image_binary = binary(image_hist_norm, thre)

#イオンの個数を計上し，その座標情報からイオン位置における電場の導出を行う
CountIons(img1)

"""
}
"""


















'''
PLOT FUNCTION
'''


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
