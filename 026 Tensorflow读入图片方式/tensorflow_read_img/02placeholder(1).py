#encoding=utf-8
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

img_name = 'img2.jpg'

def showimage_placeholder_opencv(filename):
    image = cv2.imread(filename)
    
#    Create a Tensorflow variable
    image_tensor = tf.placeholder('uint8', [None, None, 3])
    
    with tf.Session() as sess:
#        image_flap = tf.transpose(image_tensor, perm = [1,0,2])
#        sess.run(tf.global_variables_initializer())
        result = sess.run(image_tensor, feed_dict = {image_tensor:image})
        
    cv2.imshow('result', result)
    cv2.waitKey(0)

showimage_placeholder_opencv(img_name)