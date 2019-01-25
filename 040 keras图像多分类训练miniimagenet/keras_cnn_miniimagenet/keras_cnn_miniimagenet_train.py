# -*- coding: utf-8 -*-
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator,img_to_array
import matplotlib.pyplot as plt

num_classes = 3
model_name = 'miniimagenet.h5'
train_data_dir = './data/miniimagenet/train' #train文件夹下有对应为3分类名称文件夹
test_data_dir = './data/miniimagenet/val'   #
img_rows = 227
img_cols = 227

epochs = 10
# 批量大小
batch_size = 4
# 训练样本总数
nb_train_samples = 3120 #3*1040
#all num of val samples
nb_validation_samples = 780 #3*260
###################数据预处理###########################
# The data, shuffled and split between train and test sets:
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
train_datagen = ImageDataGenerator(
	rescale=1. / 255,
	horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
	train_data_dir,
	target_size=(img_rows, img_cols),
	batch_size=batch_size,
	class_mode='categorical')#多分类; 'binary')
print (train_generator)
validation_generator = test_datagen.flow_from_directory(
	test_data_dir,
	target_size=(img_rows, img_cols),
	batch_size=batch_size,
	class_mode='categorical')#多分类; 'binary')
print (validation_generator)			
#x_train = x_train.astype('float32')/255
#x_test = x_test.astype('float32')/255

# Convert class vectors to binary class matrices.
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

###################通过Keras的API定义卷机神经网络###############################
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(img_rows,img_rows,3)))  #input_shape=x_train.shape[1:]
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

# train the model using RMSprop
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#################Keras API训练模型并计算在测试数据上的准确率#######################
#hist = model.fit(x_train, y_train, epochs=10, shuffle=True)
#keras 2 fit_generator
#model.fit_generator(train_generator,
	#steps_per_epoch=nb_train_samples //batch_size,
	#epochs=epochs,
	#validation_data=test_generator,
	#validation_steps=nb_validation_samples//batch_size)
history = model.fit_generator(
		train_generator,
		steps_per_epoch=nb_train_samples//batch_size,#typically steps_per_epoch= dataset samples 3120 / batch_size 4
		epochs=epochs,#finacal epoches
		validation_data=validation_generator,
		validation_steps=nb_validation_samples//batch_size)#typically validation_steps = validation dataset samples 780 / batch_size 4
model.save(model_name)
print('Model Saved.')

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.savefig('loss.png')
#plt.show()

# evaluate
#loss, accuracy = model.evaluate(x_test, y_test)
#print("loss=")
#print(loss)
#print("acc=")
#print(accuracy)
#print('Test loss:', loss)
#print('Test accuracy:', accuracy)


