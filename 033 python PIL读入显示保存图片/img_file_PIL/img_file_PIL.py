# -*- coding: utf-8 -*-
"""
python PIL.Image
read show transpose save img
"""

import os
import numpy as np
from PIL import Image

#img_dir = "cat.jpg"
img_dir = "cat.png"
##########显示图片###########################################
img = Image.open(img_dir)
#img.show()
#PIL.Image 数据是 uinit8 型的，范围是0-255
print (img)

#RGB 转换为灰度图
gray = img.convert('L')
#gray.show()#error
#保存 PIL 图片
gray.save('gray_pil_save.png')

img_array = np.array(img)
# 也可以用 np.asarray(img) 区别是 np.array() 是深拷贝，np.asarray() 是浅拷贝
# 此时img就已经是一个 np.array 了，可以对它进行任意处理
print (img_array.shape)
print (img_array.dtype)
print (img_array)

#将numpy数组转换为 PIL 图片
#这里采用 matplotlib.image 读入png图片数组，注意这里读入的数组是 float32 型的，范围是 0-1，
#而 PIL.Image 数据是 uinit8 型的，范围是0-255，所以要进行转换
import matplotlib.image as mpimg
mimg = mpimg.imread(img_dir) # 这里读入的png数据是 float32 型的，范围是0-1
img_from_array = Image.fromarray(np.uint8(mimg*255))
#img_from_array.show()#error
#保存 PIL 图片
img_from_array.save('img_array_save.png')


