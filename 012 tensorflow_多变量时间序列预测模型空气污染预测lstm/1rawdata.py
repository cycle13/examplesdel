#coding=utf-8

#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import tensorflow as tf

from pandas import read_csv
from datetime import datetime

#load datatime
def parse(x):
	return datetime.strptime(x,'%Y %m %d %H')
dataset = read_csv('PRSA_data_2010.1.1-2014.12.31.csv',parse_dates=[['year','month','day','hour']],index_col=0)
dataset.drop('No',axis=1,inplace=True)

#specify column names
dataset.columns = ['pollution','dew','temp','press','wnd_dir','wnd_spd','snow','rain'] #'year','month','day','hour',
dataset.index.name = 'date'
#make all NA with 0
dataset['pollution'].fillna(0,inplace=True)
#drop first 24 hours
dataset = dataset[24:]
#summarize first 5 rows
print(dataset.head(10))
#save to file
dataset.to_csv('pollution.csv')


