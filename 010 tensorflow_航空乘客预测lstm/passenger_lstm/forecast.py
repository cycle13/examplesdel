# coding: utf-8

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

numpy.random.seed(7)

# load the dataset
dataframe = pandas.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)#原有两列，一列是时间，一列是乘客数量，这里利用usecols=[1],只取了乘客数量一列
dataset = dataframe.values #通过.values得到dataframe的值，返回shape是(144, 1)的数组形式
dataset = dataset.astype('float32') #把dataset里的所有数据从整型变为浮点型

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):#此处是否要去掉-1呢？待定中
		a = dataset[i:(i+look_back), 0]  #i:i+look_back代表用预测的时间点前的look_back点来作为输入，此处用                            #前1个点作为输入
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


