# -*- coding: utf-8 -*-
# -*- author：zzZ_CMing  CSDN address:https://blog.csdn.net/zzZ_CMing
# -*- 2018/06/05；11:41
# -*- python3.5
"""
olivetti Faces是纽约大学组建的一个比较小的人脸数据库。有40个人，每人10张图片，组成一张有400张人脸的大图片。
像素灰度范围在[0,255]。整张图片大小是1190*942，20行320列，所以每张照片大小是(1190/20)*(942/20)= 57*47
程序需配置h5py：python -m pip install h5py
博客地址：https://blog.csdn.net/zzZ_CMing，更多机器学习源码
"""
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD    # 梯度下降的优化器
from keras.utils import np_utils
from keras import backend as K

# 读取整张图片的数据，并设置对应标签
def get_load_data(dataset_path):
    img = Image.open(dataset_path)
    # 数据归一化。asarray是使用原内存将数据转化为np.ndarray
    img_ndarray = np.asarray(img, dtype = 'float64')/255
    # 400 pictures, size: 57*47 = 2679  
    faces_data = np.empty((400, 2679))
    for row in range(20):  
       for column in range(20):
           # flatten可将多维数组降成一维
           faces_data[row*20+column] = np.ndarray.flatten(img_ndarray[row*57:(row+1)*57, column*47:(column+1)*47])

    # 设置图片标签
    label = np.empty(400)
    for i in range(40):
        label[i*10:(i+1)*10] = i
    label = label.astype(np.int)

    # 分割数据集：每个人前8张图片做训练，第9张做验证，第10张做测试；所以train:320,valid:40,test:40
    train_data = np.empty((320, 2679))
    train_label = np.empty(320)
    valid_data = np.empty((40, 2679))
    valid_label = np.empty(40)
    test_data = np.empty((40, 2679))
    test_label = np.empty(40)
    for i in range(40):
        train_data[i*8:i*8+8] = faces_data[i*10:i*10+8] # 训练集对应的数据
        train_label[i*8:i*8+8] = label[i*10 : i*10+8]   # 训练集对应的标签
        valid_data[i] = faces_data[i*10+8]   # 验证集对应的数据
        valid_label[i] = label[i*10+8]       # 验证集对应的标签
        test_data[i] = faces_data[i*10+9]    # 测试集对应的数据
        test_label[i] = label[i*10+9]        # 测试集对应的标签
    train_data = train_data.astype('float32')
    valid_data = valid_data.astype('float32')
    test_data = test_data.astype('float32')

    result = [(train_data, train_label), (valid_data, valid_label), (test_data, test_label)]
    return result

# CNN主体
def get_set_model(lr=0.005,decay=1e-6,momentum=0.9):
    model = Sequential()
    # 卷积1+池化1
    if K.image_data_format() == 'channels_first':
        model.add(Conv2D(nb_filters1, kernel_size=(3, 3), input_shape = (1, img_rows, img_cols)))
    else:
        model.add(Conv2D(nb_filters1, kernel_size=(2, 2), input_shape = (img_rows, img_cols, 1)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 卷积2+池化2
    model.add(Conv2D(nb_filters2, kernel_size=(3, 3)))
    model.add(Activation('tanh'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  

    # 全连接层1+分类器层
    model.add(Flatten())  
    model.add(Dense(1000))       #Full connection
    model.add(Activation('tanh'))  
    model.add(Dropout(0.5))  
    model.add(Dense(40))
    model.add(Activation('softmax'))  

    # 选择设置SGD优化器参数
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)  
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model  

# 训练过程，保存参数
def get_train_model(model,X_train, Y_train, X_val, Y_val):
    model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,  
          verbose=1, validation_data=(X_val, Y_val))
    # 保存参数
    model.save_weights('model_weights.h5', overwrite=True)  
    return model  

# 测试过程，调用参数
def get_test_model(model,X,Y):
    model.load_weights('model_weights.h5')  
    score = model.evaluate(X, Y, verbose=0)
    return score  



# [start]
epochs = 35          # 进行多少轮训练
batch_size = 40      # 每个批次迭代训练使用40个样本，一共可训练320/40=8个网络
img_rows, img_cols = 57, 47         # 每张人脸图片的大小
nb_filters1, nb_filters2 = 20, 40   # 两层卷积核的数目（即输出的维度）

if __name__ == '__main__':  
    # 将每个人10张图片，按8:1:1的比例拆分为训练集、验证集、测试集数据
    (X_train, y_train), (X_val, y_val),(X_test, y_test) = get_load_data('olivettifaces.gif')

    if K.image_data_format() == 'channels_first':    # 1为图像像素深度
        X_train = X_train.reshape(X_train.shape[0],1,img_rows,img_cols)
        X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)  
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)  
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)  
        X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)  
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)  
        input_shape = (img_rows, img_cols, 1)

    print('X_train shape:', X_train.shape)
    # convert class vectors to binary class matrices  
    Y_train = np_utils.to_categorical(y_train, 40)
    Y_val = np_utils.to_categorical(y_val, 40)
    Y_test = np_utils.to_categorical(y_test, 40)

    # 训练过程，保存参数
    model = get_set_model()
    get_train_model(model, X_train, Y_train, X_val, Y_val)
    score = get_test_model(model, X_test, Y_test)

    # 测试过程，调用参数，得到准确率、预测输出
    model.load_weights('model_weights.h5')
    classes = model.predict_classes(X_test, verbose=0)  
    test_accuracy = np.mean(np.equal(y_test, classes))
    print("last accuarcy:", test_accuracy)
    for i in range(0,40):
        if y_test[i] != classes[i]:
            print(y_test[i], 'error predict', classes[i])
