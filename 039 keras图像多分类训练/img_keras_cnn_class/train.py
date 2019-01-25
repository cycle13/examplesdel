# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator,img_to_array
import matplotlib.pyplot as plt

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150,3)))
# filter大小3*3，数量32个，原始图像大小3,150,150
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))   #matt,几个分类就要有几个dense
model.add(Activation('softmax'))#matt,多分类

# 二分类
#model.compile(loss='binary_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])

# 多分类
# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
model.compile(loss='categorical_crossentropy',# matt，多分类，不是binary_crossentropy
              optimizer='rmsprop',#opt,#'rmsprop',
              metrics=['accuracy'])
# 优化器rmsprop：除学习率可调整外，建议保持优化器的其他默认参数不变

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        './data/train', 
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=4,#32
        class_mode='categorical')# matt，多分类

validation_generator = val_datagen.flow_from_directory(
        './data/val',
        target_size=(150, 150),
        batch_size=4,#32
        class_mode='categorical')# matt，多分类
#class_mode='binary'

#model.fit_generator(
	#train_generator,
	#samples_per_epoch=2000,
	#nb_epoch=50,
	#validation_data=validation_generator,
	#nb_val_samples=800)
# samples_per_epoch，相当于每个epoch数据量峰值，每个epoch以经过模型的样本数达到samples_per_epoch时，记一个epoch结束
history = model.fit_generator(
		train_generator,
		steps_per_epoch=100,#typically steps_per_epoch= dataset samples 400 / batch_size 4
		epochs=5000,
		validation_data=validation_generator,
		validation_steps=25)#typically validation_steps = validation dataset samples 100 / batch_size 4
model.save_weights('./model_weights.h5')  
#model.save(model_file)#elesun
print('Model weights Saved.')#elesun
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



