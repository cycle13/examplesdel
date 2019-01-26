
from keras.layers import Input,concatenate,Dense
from keras.models import Sequential, Model

# merge samples, two input must be same shape
inp1 = Input(shape=(10,32))
inp2 = Input(shape=(10,32))
cc1 = concatenate([inp1, inp2],axis=0) # Merge data must same row column
output = Dense(30, activation='relu')(cc1)
model = Model(inputs=[inp1, inp2], outputs=output)
model.summary()

# merge row must same column size
inp1 = Input(shape=(20,10))
inp2 = Input(shape=(32,10))
cc1 = concatenate([inp1, inp2],axis=1)
output = Dense(30, activation='relu')(cc1)
model = Model(inputs=[inp1, inp2], outputs=output)
model.summary()

# merge column must same row size
inp1 = Input(shape=(10,20))
inp2 = Input(shape=(10,32))
cc1 = concatenate([inp1, inp2],axis=2)
output = Dense(30, activation='relu')(cc1)
model = Model(inputs=[inp1, inp2], outputs=output)
model.summary()
