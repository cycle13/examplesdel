# -*- coding: utf-8 -*-
import os
import random
import cv2
import numpy as np
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator,img_to_array
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.optimizers import SGD

#img_file = './img/female1.jpg'
#img_file = './img/female2.jpg'
#img_file = './img/male1.jpg'
img_file = './img/male2.jpg'

IMAGE_SIZE = 182
# 训练图片大小

epochs = 10#150#原来是50
# 遍历次数

batch_size = 8#32
# 批量大小

nb_train_samples = 896*2#512*2
# 训练样本总数

nb_validation_samples = 128*2#128*2
# 测试样本总数

train_data_dir = './data/train/'
validation_data_dir = './data/val/'
# 样本图片所在路径

MODEL_PATH = 'model_weights.h5'
# 模型存放路径

class Dataset(object):

    def __init__(self):
        self.train = None
        self.valid = None


    def read(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='binary')

        validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='binary')

        self.train = train_generator
        self.valid = validation_generator


class Model(object):



    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        
        
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        self.model.summary()
        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    def train(self, dataset, batch_size=batch_size, nb_epoch=epochs):
        self.model.fit_generator(dataset.train,
                                 steps_per_epoch=nb_train_samples // batch_size,
                                 epochs=epochs,
                                 validation_data=dataset.valid,
                                 validation_steps=nb_validation_samples//batch_size)


    def save(self, file_path=MODEL_PATH):
        print('Model Saved.')
        self.model.save_weights(file_path)

    def load(self, file_path=MODEL_PATH):
        print('Model Loaded.')
        self.model.load_weights(file_path)

    def predict(self, image):
        # 预测样本分类

        img = cv2.imread(image)
        img = cv2.resize(img,(IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype('float32')/255.0
        img = img_to_array(img)
        img = np.expand_dims(img,axis=0)#reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        #print (img)
        print (img.shape)
        #image process

        result = self.model.predict(img)
        print "model.predict = ",result
        #prob

        result = self.model.predict_classes(img)
        print "model.predict_classes = ",result
        #class 0 or 1

        return result[0]

    def evaluate(self, dataset):
        # 测试样本准确率
        score = self.model.evaluate_generator(dataset.valid,steps=10)
        print("val samples %s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

if __name__ == '__main__':
    dataset = Dataset()
    dataset.read()


    model = Model()
    model.load()#must make model firstly
    #model.train(dataset)
    #model.evaluate(dataset)
    model.predict(img_file)
    #model.save()

