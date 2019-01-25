#coding=utf-8  
from keras.models import Sequential  
from keras.layers import Dense,Flatten,Dropout  
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical  

import numpy as np  
from keras.utils.vis_utils import plot_model #elesun plt

def ZF_Net():
    model = Sequential()  
    model.add(Conv2D(96,(7,7),strides=(2,2),input_shape=(224,224,3),padding='valid',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
    model.add(Conv2D(256,(5,5),strides=(2,2),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
    model.add(Flatten())  
    model.add(Dense(4096,activation='relu'))  
    model.add(Dropout(0.5))  
    model.add(Dense(4096,activation='relu'))  
    model.add(Dropout(0.5))  
    model.add(Dense(1000,activation='softmax'))  
    return model

if __name__ == "__main__":
	model = ZF_Net()
	model.summary()#elesun
	plot_model(model,to_file='plt_ZF_Net.png',show_shapes=False,show_layer_names=False) #elesun plt	
	#model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])  	  
	#model.fit(train_x,train_y,validation_data=(valid_x,valid_y),batch_size=20,epochs=20,verbose=2)    
	#print model.evaluate(test_x,test_y,batch_size=20,verbose=2)  





