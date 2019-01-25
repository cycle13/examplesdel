#encoding=utf-8
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

img_name = 'img1.jpg'

def showimage_gfile(filename):
    #读物文件
    image_raw = tf.gfile.FastGFile(filename, 'rb').read()
    #图像解码
    image_data = tf.image.decode_jpeg(image_raw)
    #改变图像数据的类型
    #image_show = tf.image.convert_image_dtype(image_data, dtype = tf.uint8)     
    plt.figure(1) 	
    with tf.Session() as sess:
        print(type(image_raw)) # bytes
        print(type(image_data)) # Tensor
        print(type(image_data.eval())) # ndarray
        plt.imshow(image_data.eval())
        plt.show()


showimage_gfile(img_name)