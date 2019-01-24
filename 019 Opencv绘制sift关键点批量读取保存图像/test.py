# -*- coding: utf-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

dir = "C://Users//Administrator//netlearn//SIFT_BFmatcher//SRAD2018_TRAIN_001//RAD_206482464212530//"

sift = cv2.xfeatures2d.SIFT_create()

def draw_radar_sift(dir):
	file_list = os.listdir(dir)
	print(file_list)
	for filename in file_list:
		path = ''
		path = dir+filename
		gray = cv2.imread(path,0) #读取灰度图
		#gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #灰度处理图像
		kp, des = sift.detectAndCompute(gray,None)   #des是描述子
		img2 = cv2.drawKeypoints(gray,kp,gray,color=(255,0,255)) #画出特征点，并显示为红色圆圈
		#cv2.imshow("point", img2)
		cv2.imwrite(path, img2)
		print ("%s has been draw sift!"%filename)
		cv2.waitKey(0)		
        

if __name__ == '__main__':
   draw_radar_sift(dir)

'''
import numpy as np
import cv2
from matplotlib import pyplot as plt

imgname1 = 'RAD_206482464212530_038.png'
imgname2 = 'RAD_206482464212530_054.png'

sift = cv2.xfeatures2d.SIFT_create()

img1 = cv2.imread(imgname1,0)
#gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #灰度处理图像
kp1, des1 = sift.detectAndCompute(img1,None)   #des是描述子

img2 = cv2.imread(imgname2,0)
#gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)#灰度处理图像
kp2, des2 = sift.detectAndCompute(img2,None)  #des是描述子

hmerge = np.hstack((img1, img2)) #水平拼接
cv2.imshow("gray", hmerge) #拼接显示为gray
#cv2.imwrite("gray", hmerge)
cv2.waitKey(0)

img3 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255)) #画出特征点，并显示为红色圆圈
img4 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255)) #画出特征点，并显示为红色圆圈
hmerge = np.hstack((img3, img4)) #水平拼接
cv2.imshow("point", hmerge) #拼接显示为gray
#cv2.imwrite("point", hmerge)
cv2.waitKey(0)
# BFMatcher解决匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
# 调整ratio
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

#img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)
img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good[:10],None,flags=2)
cv2.imshow("BFmatch", img5)
#cv2.imwrite("BFmatch", img5)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
