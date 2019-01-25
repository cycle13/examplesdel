# -*- coding: utf-8 -*-
"""
matplotlib scipy
read show transpose save img
"""

import os
#用于显示图片
import matplotlib.pyplot as plt
#用于读取图片
import matplotlib.image as mpimg
from scipy import misc
import numpy as np

img_dir = "cat.jpg"
#img_dir = "cat.png"
##########显示图片###########################################
img = mpimg.imread(img_dir) # 读取和代码处于同一目录下的图片
# 此时img就已经是一个 np.array 了，可以对它进行任意处理
print (img.shape) #(50, 50,3)
print (img.dtype)
print (img)
plt.imshow(img) #显示图片
plt.title('show img')
plt.axis('off') #不显示坐标轴
plt.show()
##########RGB转为灰度图###########################################
def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
gray = rgb2gray(img)  
plt.imshow(gray)
plt.title('hot gray img')
plt.show()
# 此时会发现显示的是热量图，不是我们预想的灰度图，可以添加 cmap 参数，有如下几种添加方法

#也可以用 plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.imshow(gray, cmap='Greys_r')
plt.title('gray img')
plt.axis('off')
plt.show()
##########放缩图片###########################################
img_new_sz = misc.imresize(img,0.5) # 第二个参数如果是整数，则为百分比，如果是tuple，则为输出图像的尺寸
plt.imshow(img_new_sz)
plt.title('resize img')
plt.axis('off')
plt.show()
##########保存图片###########################################
#保存 matplotlib 画出的图像，适用于保存任何 matplotlib 画出的图像，相当于一个screencapture
plt.imshow(img_new_sz)
plt.title('save img')
plt.axis('off')
plt.savefig('img_plt_save.png')
print("img_plt_save.png saved")
#将array保存为图像
misc.imsave('img_sci_save.png',img_new_sz)
print("img_sci_save.png saved")

