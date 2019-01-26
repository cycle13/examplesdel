# -*- coding: utf-8 -*-
'''
keras getlayer
#获得某一层的权重和偏置
#获得某一层的featuremap shape
env:Linux ubuntu 4.4.0-31-generic x86_64 GNU;python 2.7;tensorflow1.10.1;Keras2.2.4
'''
from keras.models import Sequential,Model
from keras.layers import Dense
import numpy as np
import os
from keras.models import load_model

model_dir = 'model'
model_file = os.path.join(model_dir, 'model_getlayer.h5')

#假设训练和测试使用同一组数据
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))
print 'data.shape=',data.shape
print 'labels.shape=',labels.shape

'''
#define model network
model = Sequential()
model.add(Dense(32,activation="relu",input_dim=100,name="Dense_0"))
model.add(Dense(16,activation="relu",name="Dense_1"))
model.add(Dense(1, activation='sigmoid',name="Dense_2"))
model.summary()
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
#train and save model
model.fit(data,labels,epochs=10,batch_size=32)
print('save the trained model')
if not os.path.exists(model_dir):
	os.mkdir(model_dir)
model.save(model_file)#elesun
'''

#load  model and test
print('load the trained model')
if not os.path.isfile(model_file):
    print(model_file+" not exist!")
    exit(0)
model = load_model(model_file)
print 'model.input=',model.input
print 'model.input.name=',model.input.name
print 'model.input.shape=',model.input.shape
print 'model.output=',model.output
print 'model.output.name=',model.output.name
print 'model.output.shape=',model.output.shape
#取某一层的输出为输出新建为model，采用函数模型
dense1_layer_model = Model(inputs=model.input,outputs=model.get_layer('Dense_1').output)
dense1_output = dense1_layer_model.predict(data)
print 'dense1_output.shape=',dense1_output.shape
print 'dense1_output featuresmap=\n',dense1_output

#获得某一层的权重和偏置
weight_Dense_1,bias_Dense_1 = model.get_layer('Dense_1').get_weights()
print 'weight_Dense_1.shape=',weight_Dense_1.shape
print 'bias_Dense_1.shape=',bias_Dense_1.shape

