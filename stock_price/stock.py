import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import warnings
import base64
import io
import os, fnmatch
from random import choice

import sklearn
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
#from keras.layers import Dense,Dropout,LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM

rcParams['figure.figsize'] = 20,10
df3 = pd.read_csv('Apple.csv')
df3.head(10)
df3.shape
df3.columns

warnings.filterwarnings('ignore')



def graph():
	
	plt.figure(figsize=(16,6))
	plt.plot(df3['close'], label = 'Closing Price History')
	plt.savefig('static/graph.png')
	

new_data = pd.DataFrame(df3, columns = ['date','close'])
new_data.sort_index().head()

data = df3.sort_index(ascending = True, axis=0)
new_data = pd.DataFrame(index = range(0,len(df3)), columns = ['date','close'])

for i in range(0, len(df3)):
    new_data['date'][i] = data['date'][i]
    new_data['close'][i] = data['close'][i]

new_data.index = new_data.date
new_data.drop('date', axis=1, inplace=True)
new_data.head(10)

#creating train and test sets
dataset = new_data.values
train = dataset[0:460,:]
valid = dataset[460:,:]
print(new_data.shape)
print(train.shape)
print(valid.shape)

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

#scaled_data.shape
scaled_data

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# Creating and fitting the model

model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(units = 50))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(x_train, y_train, epochs=1, batch_size = 1, verbose = 2)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)
rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
rms

train = new_data[:460]
new_data.shape
valid = new_data[460:]
valid['Predictions'] = closing_price
#print(valid[['close','Predictions']])
def gp():
	plt.plot(train['close'])
	plt.plot(valid[['close','Predictions']])
	plt.savefig('static/graph1.png')

valid

ypred = model.predict(X_test, steps = 2)
ypred1 = scaler.inverse_transform(ypred)
ypred1.shape

print(valid)
def values(b):
	c=42+b
	return ypred1[42:c]
#plt.show()

# predictions = lstm.predict_sequences_multiple(model,X_test,50)
# lstm.plot_results_multiple(predictions,y_test,50)

# model.predict(X_test,steps=1)
# X_test.shape
# print(valid[['close','Predictions']])
# plt.figure(figsize=(15,5))
# plt.plot(train['close'])
# plt.plot(valid[['close','Predictions']])