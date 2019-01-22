# -*- coding: utf-8 -*-

#������Ӧ�Ŀ�
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

#�����ݴ洢Ϊ��������һ�������indλ�ô洢tʱ�̵�ֵ����һ������洢t+1ʱ�̵�ֵ
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

#��ȡ����
dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

#�鿴���ݼ�
print('2nd num of dataset:\n',dataset[0:2])
print('length of dataset:',len(dataset))

plt.plot(dataset)
plt.show()

#LSTM���������ݵĹ�ģ�����У��ر�����ʹ��sigmoid��Ĭ�ϣ���tanh�����ʱ��
#���������µ�����0��1�ķ�Χ��Ҳ��Ϊ��׼����������һ�ֺܺõ�������

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# ����ѵ��������Լ�
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

print('split dataset,len(train),len(test)',train.shape,test.shape)

# ����[t,t+look_back]ʱ������t+look_backʱ�̵���������
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

print(trainX[:2])
print(trainY[:2])

# ���ݱ�Reshape�� [samples, time steps, features]�����Ƿ���LSTM��shape
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print('input data with label,len(trainX),len(testX)',trainX.shape,testX.shape)

#����LSTM����
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))

#����ѵ��LSTM����
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=1)

#��ӡģ��
model.summary()

#����ģ��
SVG(model_to_dot(model,show_shapes=True).create(prog='dot', format='svg'))


# ʹ����ѵ����ģ�ͽ���Ԥ��
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Ԥ���ֵ��[0,1]�����ı�׼�����ݣ���Ҫ����ֵת����ԭʼֵ
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# ����Ԥ��ľ��������
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# ��ͼ����ѵ�����ݵ�Ԥ��
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict


# ��ͼ���Բ������ݵ�Ԥ��
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
testPredictPlot[len(trainPredict)+look_back:len(dataset)-1, :] = testPredict

# ��ʾͼƬ
plt.plot(scaler.inverse_transform(dataset),color='blue',label='Raw data')
plt.plot(trainPredictPlot,color='red',label='Train process')
plt.plot(testPredictPlot,color='green',label='Test process')

#������ͼ����ʾ��ǩ
leg = plt.legend(loc='best', ncol=1, fancybox=True)
leg.get_frame().set_alpha(0.5)

plt.show()


#�������ݵ����һ������û��Ԥ��,���ﲹ��
finalX = numpy.reshape(test[-1], (1, 1, testX.shape[1]))

#Ԥ��õ���׼������
featruePredict = model.predict(finalX)

#����׼������ת��Ϊ����
featruePredict = scaler.inverse_transform(featruePredict)

#ԭʼ������1949-1960�������,��һ������1961��1�·�
print('model forcast 1961.1 passenger num :',featruePredict)
