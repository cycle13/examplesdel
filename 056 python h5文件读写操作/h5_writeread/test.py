# -*- coding: utf-8 -*-
import h5py
import numpy as np
 
#HDF5的写入：
imgData = np.random.rand(60,500,500,3) #frames,width,hight,channel
f = h5py.File('HDF5_FILE.h5','w')   #创建一个h5文件，文件指针是f
f['data'] = imgData                 #将数据写入文件的主键data下面
f['labels'] = np.array([0,1,2,3,4,5,6,7,8,9])            #将数据写入文件的主键labels下面
f['date'] = ('20181220','20181221','20181222','20181223','20181224','20181225')
f.close()                           #关闭文件
 
#HDF5的读取：
f = h5py.File('HDF5_FILE.h5','r')   #打开h5文件
# 可以查看所有的主键
for key in f.keys():
    print 'f[key].name = ',f[key].name
    print 'f[key].shape = ',f[key].shape
    print 'f[key].value = \n',f[key].value
a = f['date'][:]                    #取出主键为date的所有的键值
print 'a = ',a
b = f['data'][0,:,:,0]              #取出主键为data的some的键值
print 'b = \n',b
