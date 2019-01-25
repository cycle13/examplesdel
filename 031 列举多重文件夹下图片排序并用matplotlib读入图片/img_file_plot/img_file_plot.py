# -*- coding: utf-8 -*-
"""
列举文件夹下文件夹下的图片列表并排序
Python通过matplotlib实现读取图片
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img_dir = "./data/" #顶级文件夹路径


#列举顶级文件夹下的文件夹列表
file_list = os.listdir(img_dir)
#文件夹按指定字符大小排序，RAD_206582404232531_50x50，取中间的206582404232531
file_list.sort(key=lambda x:int(x[4:-6]))#从文件夹名称第4个开始（数组起始序号为0，包含第4个），到倒数第6个数字结尾（不包含）
#打印排序后的文件夹列表
print(file_list)
#文件夹对应一个序号，初始为0
filecnt = 0
for filename in file_list:	
	path = ''
	path = img_dir+filename
	#列举文件夹下的图片列表
	img_list = os.listdir(path)
	#图片文件按指定字符大小排序，RAD_206582404232531_001.png，取中间的001
	img_list.sort(key=lambda x:int(x[20:-4]))##从图片名称第20个开始（数组起始序号为0，包含第20个），到倒数第4个数字结尾（不包含）'.'为分界线，按照‘.’左边的数字从小到大排序
	#打印排序后的图片列表
	print(img_list)
	#图片对应一个序号，初始为0
	imgcnt = 0
	for imgname in img_list:
		path = ''
		path = img_dir+filename+"/"+imgname
		img = mpimg.imread(path) # 读取和代码处于同一目录下的图片
		# 此时 img 就已经是一个 np.array 了，可以对它进行任意处理
		print (img.shape) #(50, 50)
		print (img.dtype)
		print (img)
		#一个图片对应一个序号
		imgcnt = imgcnt + 1
	#一个文件夹对应一个序号
	filecnt = filecnt + 1
