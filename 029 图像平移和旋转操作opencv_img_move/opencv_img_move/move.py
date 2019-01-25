
import cv2
import numpy as np
 
dx = 100
dy = 100
 
img = cv2.imread('cat.jpg', 1)
rows,cols,channel = img.shape
cv2.imshow('source',img)
 
matShift = np.float32([[1,0,dx],[0,1,dy]])
dst = cv2.warpAffine(img,matShift,(cols,rows))
 
cv2.imshow('move',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
