# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 12:22:31 2021

@author: akira
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pylab import *



datafile='20210303_3_50_i'
threshold = 400
#イオンの数とパラメータを推測するために使用
width=2
#イオン列付近の画像を切り取るために使用

def read_data(myfile):
    f=open(myfile,'r')
    datastr=f.readlines()
    f.close()
    tmax=len(datastr)
    xmax=len(datastr[0].split('\t'))
    img_data=zeros([tmax,xmax])
    for i in range(tmax):
        this_line=datastr[i].split('\t')
        for j in range(xmax):
            img_data[i][j]=int(this_line[j])
    return(img_data)

def sum_over_time(img_data):
    x_prof=zeros(len(img_data[0]))
    for i in range(len(img_data)):
        x_prof=x_prof+img_data[i]
    return(x_prof)
    
def func(x, *params):
#paramsの長さでフィッティングする関数の数を判別。
    num_func = int(len(params)/3)
#ガウス関数にそれぞれのパラメータを挿入してy_listに追加。
    y_list = []
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(3*i,3*(i+1),1))
        intensity = params[int(param_range[0])]
        X0 = params[int(param_range[1])]
        HWHM = params[int(param_range[2])]
        y = y + intensity * HWHM**2 / ((x-X0)**2 + HWHM**2)
        y_list.append(y)
#y_listに入っているすべてのガウス関数を重ね合わせる。
    y_sum = np.zeros_like(x)
    for i in y_list:
        y_sum = y_sum + i
#最後にバックグラウンドを追加。
    y_sum = y_sum + params[-1]
    return y_sum

def fit_plot(x, *params):
    num_func = int(len(params)/3)
    y_list = []
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(3*i,3*(i+1),1))
        intensity = params[int(param_range[0])]
        X0 = params[int(param_range[1])]
        HWHM = params[int(param_range[2])]
        y = y +intensity * HWHM**2 / ((x-X0)**2 + HWHM**2) + params[-1]
        y_list.append(y)

        return y_list

def find_initial_value(x,y,threshold):
    x_i=[]
    z = []
    for i in range(len(x)-1):
        if (y[i]-threshold)*(y[i+1]-threshold) < 0:
            x_i.append(x[i])
    for i in range(int(len(x_i)/2)):
        z.append(y[int((x_i[2*i]+x_i[2*i+1])/2)])
        z.append(int((x_i[2*i]+x_i[2*i+1])/2))
        z.append((x_i[2*i+1]-x_i[2*i]))
    return z

def find_max_intense(img_data):
    data_MAX=0
    for i in list(range(len(img_data))):
        for j in list(range(len(img_data[0]))):
            if(data_MAX<abs(img_data[i][j])):
                data_MAX=img_data[i][j]
                center=i
    return center

img_data=[]
img_data=read_data(datafile)
center = find_max_intense(img_data)
#中心を探す

img_data2=zeros([2*width,len(img_data)])
for i in list(range(center-width,center+width)):
    for j in list(range(len(img_data))):
        img_data2[i-(center-width)][j]=img_data[i][j]
#中心付近のデータのみ取得
    
x_prof=sum_over_time(img_data2)
X=[]
Y=[]
for i in list(range(len(x_prof))):
    X.append(i)
    Y.append(x_prof[i])
   
gauss_total = find_initial_value(X,Y,threshold)
print(gauss_total)
background=0
gauss_total.append(background)
#初期値リストの作成 
        
popt, pcov = curve_fit(func, X, Y, p0=gauss_total)
perr = np.sqrt(np.diag(pcov))
#フィッテイング

fit = func(X, *popt)
plt.scatter(X, Y, s=10)
plt.plot(X, fit , ls='-', c='black', lw=1)
plt.axhline(y=threshold, color='red',linewidth=1)

y_list = fit_plot(X, *popt)
baseline = np.zeros_like(X) + popt[-1]


name = ['intensity','position','HWHM    ']
print("initial parameter\toptimized parameter")
for i, v  in enumerate(gauss_total):
    if i==len(gauss_total)-1:
        print('\n' + 'background' + '\t' + str(v)+ '\t' + str(popt[i]) + ' ± ' + str(perr[i]))
    elif i%3==0:
        print('No.' + str(int(i/3)+1))
    if i!=len(gauss_total)-1:
        print( name[i%3] + '\t' + str(v)+ '\t' + str(popt[i]) + ' ± ' + str(perr[i]))
# 結果の表示

plt.tick_params(labelsize=15)
plt.xlabel('pixel number',fontsize=18)
plt.ylabel('gray scale',fontsize=18)
