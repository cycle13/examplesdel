""" This script demonstrates the use of a convolutional LSTM network.

This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import os

import preprocessing
import conv_lstm_radar

mode = 'train' # train or test
n_samples=9
n_frames=61
which = 1#<n_samples,which samples you select to predict elesun
batch_size= 4#8
epochs= 100#1000
model_dir = 'model'
out_dir = 'output'
model_file = os.path.join(model_dir, 'model_train_radar.h5')

# preprocessing data
print("######preprocessing data######")
now_radar_mat, next_radar_mat = preprocessing.generate_radar_data(n_samples,n_frames)
print(now_radar_mat.shape)
print(now_radar_mat.dtype)
#print(now_radar_mat)
print(next_radar_mat.shape)
print(next_radar_mat.dtype)
#print(next_radar_mat)

print('########'+mode+'########')
if(mode=='train'):
	model = conv_lstm_radar.network()
	early_stopping = EarlyStopping(monitor='val_loss',patience=6)#by elesun EarlyStopping
	history = model.fit(now_radar_mat[::,:61,::,::,::], next_radar_mat[::,:61,::,::,::], batch_size=batch_size,epochs=epochs, validation_split=0.05,callbacks=[early_stopping]) #by elesun EarlyStopping
	#history = model.fit(now_radar_mat[::,:61,::,::,::], next_radar_mat[::,:61,::,::,::], batch_size=batch_size,epochs=epochs, validation_split=0.05)
	if not os.path.exists(model_dir):
	    os.mkdir(model_dir)
	model.save(model_file)#elesun
	print('Model Saved.')#elesun
	# plot history
	plt.plot(history.history['loss'], label='train')
	plt.plot(history.history['val_loss'], label='test')
	plt.legend()
	plt.savefig('loss.png')
	plt.show()
elif(mode=='test'):
	# load the trained model
	print('#load the trained model:')
	if not os.path.isfile(model_file):
	    print(model_file+" not exist!")
	    exit(0)
	model = load_model(model_file)
else :
	print('#there is no your mode! tips:mode=train or test')
	exit(0)


# evaluate
print("######model evaluate######")
#loss, accuracy = model.evaluate(steps=10)#now_radar_mat[::,31:,::,::,::],next_radar_mat[::,31:,::,::,::])#(x_test, y_test) elesun
loss = model.evaluate(now_radar_mat[::,31:,::,::,::],next_radar_mat[::,31:,::,::,::],batch_size=4,verbose=0)
#print('Test loss:', loss)
#print 'Test accuracy:', accuracy
print model.metrics_names,loss

# Testing the network on one movie
# feed it with the first 7 positions and then
# predict the new positions
print("######model predict######")
track = now_radar_mat[which][:31, ::, ::, ::]

for j in range(30):#predict future frames elesun
    new_pos = model.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)


# And then compare the predictions
# to the ground truth
track2 = now_radar_mat[which][::, ::, ::, ::]
for i in range(61):#frames all elesun
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    if i >= 31:#predict frames
        ax.text(1, 3, 'Predictions !', fontsize=20, color='g')
    else:#already frames
        ax.text(1, 3, 'before frames', fontsize=20, color='g')

    toplot = track[i, ::, ::, 0]

    plt.imshow(toplot,cmap='Greys_r')
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20, color='g')

    toplot = track2[i, ::, ::, 0]
    if i >= 2:
        toplot = next_radar_mat[which][i - 1, ::, ::, 0]

    plt.imshow(toplot,cmap='Greys_r')
    if not os.path.exists(out_dir):
           os.mkdir(out_dir)
    plt.savefig(out_dir+'/'+'%i_contrast.png' % (i + 1))
