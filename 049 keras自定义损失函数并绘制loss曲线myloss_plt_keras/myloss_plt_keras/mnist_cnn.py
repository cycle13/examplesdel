'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt

batch_size = 32#128
num_classes = 10
epochs = 100#12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print ('model.metrics_names = ',model.metrics_names)
history1 = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
print('history1.history = ',history1.history)
print('history1.epoch = ',history1.epoch)			  
#custom loss
def mycrossentropy(y_true, y_pred, e=0.1):
    return (1-e)*K.categorical_crossentropy(y_pred,y_true) + e*K.categorical_crossentropy(y_pred, K.ones_like(y_pred)/num_classes)

model.compile(loss=mycrossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print (model.metrics_names)
history2 = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
print('history2.history = ',history2.history)
print('history2.epoch = ',history2.epoch)

#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

# plot history
plt.title("model performace")
plt.plot(history1.epoch,history1.history['loss'], label='train_loss')
plt.plot(history1.epoch,history1.history['val_loss'], label='test_loss')
plt.plot(history1.epoch,history1.history['acc'], label='train_acc')
plt.plot(history1.epoch,history1.history['val_acc'], label='test_acc')

plt.plot(history2.epoch,history2.history['loss'], label='my_train_loss')
plt.plot(history2.epoch,history2.history['val_loss'], label='my_test_loss')
plt.plot(history2.epoch,history2.history['acc'], label='my_train_acc')
plt.plot(history2.epoch,history2.history['val_acc'], label='my_test_acc')

plt.ylabel("loss or acc")
plt.xlabel("epochs")
plt.legend()
plt.savefig('history.png')
plt.show()

