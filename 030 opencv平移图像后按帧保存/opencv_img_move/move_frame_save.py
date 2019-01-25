
import cv2
import numpy as np
 
dx = 1
dy = 1
borderColor=(255,255,255) #white color
cnt = 500
out_dir = "out"
 
img = cv2.imread('RAD_206582404232538_014.png',1)
rows,cols,channel = img.shape
print img.shape
#cv2.imshow('source',img)

for n in range(cnt):
	path =""
	matShift = np.float32([[1,0,dx*n],[0,1,dy*n]])
	dst = cv2.warpAffine(img,matShift,(cols,rows),borderMode=cv2.BORDER_CONSTANT,borderValue=borderColor)
	path = out_dir+"/frame_"+"%02d"%n+".png"
	cv2.imwrite(path,dst)        
	print ("%s has been saved!"%path)

