# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 12:52:10 2025

@author: ismt
"""

from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#make a original data
series = np.sin((0.1*np.arange(400))**2)

#a time series from x(t) = sin(wt**2)

#visualize 
plt.plot(series)
plt.show()

#build data_Set
T = 10
D = 1
X = []
Y = []

for t in range(len(series)-T):
    x = series[t:t+T]
    X.append(x)
    y = series[t+T]
    Y.append(y)
    
#X = np.array(X).reshape(-1,T)
X = np.array(X)
Y = np.array(Y)
N = len(X)
print(f"X.shape = {X.shape}\nY.shape = {Y.shape}")

#AutoRegressive Linear Model
i = Input(shape=(T,))
x = Dense(1)(i)
model=Model(i, x)
model.compile(loss="mse",
              optimizer=Adam(lr=0.01))

history = model.fit(X[:-N//2], Y[:-N//2],
                    epochs=80,
                    validation_data=(X[-N//2:], Y[-N//2:]),
                    )


#plot loss per iteration
plt.figure()
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")

#evaluate FALSE
outputs = model.predict(X)
print(outputs.shape)
predictions = outputs[:,0]

plt.figure()
plt.plot(Y, label="targets")
plt.plot(predictions, label='predictions')
plt.legend()
plt.show()

#evaluate TRUE
validation_target = Y[-N//2:]
validation_predictions=[]

#index of first validation_input = last train data idx
i = -N//2
while len(validation_predictions) < len(validation_target):
    p = model.predict(np.expand_dims(X[i], axis=0))[0,0]
    i+=1
    validation_predictions.append(p)

#multistep_forecast
validation_target = Y[-N//2:]
validation_predictions=[]

last_x = X[-N//2] #1D array of len T

while len(validation_predictions) < len(validation_target):
    p = model.predict(np.expand_dims(last_x, axis=0))[0,0]
    validation_predictions.append(p)
    
    #update last_x
    last_x = np.roll(last_x, -1)
    last_x[-1] = p
    

plt.figure()
plt.plot(validation_target, label="targets")
plt.plot(validation_predictions, label='predictions')
plt.legend()
plt.show()


#try RNN/LSTM model
X = np.expand_dims(X, axis=-1) # N*T*D 3D data we need model trained thisway

#MAKE RNN MODEL
i = Input(shape=(T,D))
#x = SimpleRNN(10)(i)
x = LSTM(10)(i)

x = Dense(1)(x)
model = Model(i, x)

model.compile(loss="mse",
              optimizer=Adam(lr=0.01))

history = model.fit(X[:-N//2], Y[:-N//2],
                    batch_size=1,
                    epochs=200,
                    validation_data=(X[-N//2:], Y[-N//2:]))




#plot loss per iteration
plt.figure()
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")    

# #evaluate FALSE
# outputs = model.predict(X)
# print(outputs.shape)
# predictions = outputs[:,0]

# plt.figure()
# plt.plot(Y, label="targets")
# plt.plot(predictions, label='predictions')
# plt.legend()
# plt.show()


#evaluate True
forecast=[]
input_ = X[-N//2]

while len(forecast) < len(validation_target):
    p = model.predict(np.expand_dims(input_, axis=0))[0, 0]
    forecast.append(p)
    
    input_ = np.roll(input_,-1)
    input_[-1] = p


    
plt.figure()
plt.plot(validation_target, label="targets")
plt.plot(forecast, label='predictions')
plt.legend()
plt.show()




