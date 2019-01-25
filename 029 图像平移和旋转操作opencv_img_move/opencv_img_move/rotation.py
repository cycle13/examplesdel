import cv2
import numpy as np
 
img = cv2.imread('cat.jpg', 1)
rows,cols,channel = img.shape
cv2.imshow('source',img)
 
M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
dst = cv2.warpAffine(img,M,(cols,rows))
 
cv2.imshow('rotation',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
