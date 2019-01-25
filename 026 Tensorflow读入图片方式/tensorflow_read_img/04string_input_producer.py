#encoding=utf-8
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

img_name = 'img1.jpg'

def showimage_string_input(filename):
#    函数接受文件列表，如果是文件名需要加[]
    file_queue = tf.train.string_input_producer([filename])
    
#    定义读入器，并读入文件缓存器
    image_reader = tf.WholeFileReader()
    _, image = image_reader.read(file_queue)
    image = tf.image.decode_jpeg(image)
    
    with tf.Session() as sess:
#       初始化协同线程
        coord = tf.train.Coordinator()
#       启动线程
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)  
        result = sess.run(image)
        coord.request_stop()
        coord.join(threads)
        image_uint8 = tf.image.convert_image_dtype(image, dtype = tf.uint8)
        plt.imshow(image_uint8.eval())

    cv2.imshow('result', result)
    cv2.waitKey(0)


showimage_string_input(img_name)