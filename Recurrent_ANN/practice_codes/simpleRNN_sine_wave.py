# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:09:48 2025

@author: ismt
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

series = np.sin(0.1*np.arange(200)) + np.random.randn(200)*0.1

#plot it
plt.plot(series)
plt.show()

T=10 #len of X
D = 1 #features
X = []
Y = []

for t in range(len(series)-T):
    x = series[t:t+T]
    X.append(x)
    y = series[t+T]
    Y.append(y)
    
X = np.array(X).reshape(-1, T, 1) #now data should be N*T*D
Y = np.array(Y)
N=len(X)
print(f"X.shape is {X.shape}\nY.shape is {Y.shape}")

#try build autoregressive RNN model
i = Input(shape=(T, 1))
x = SimpleRNN(5, activation="tanh")(i) #default tanh
x = Dense(1)(x)
model=Model(i, x)

model.compile(loss="mse",
              optimizer=Adam(lr=0.1))

#train
history = model.fit(
    X[:-N//2], Y[:-N//2],
    epochs=80,
    batch_size=1,
    validation_data=(X[-N//2:], Y[-N//2:]),
    
    )

loss = history.history["loss"]
val_loss =  history.history["val_loss"]
plt.figure()
plt.plot(loss, label="loss")
plt.plot(val_loss, label="val_loss")
plt.legend()

plt.show()


#wrong prediction
validation_target = Y[-N//2:]
validation_predictions = []
#idx of first input 
i = -N//2

while len(validation_predictions) < len(validation_target):
    p = model.predict(np.expand_dims(X[i], axis=0))[0,0]
    i+=1
    validation_predictions.append(p)

plt.title("wrong")
    
plt.plot(validation_target, label="forecast target")
plt.plot(validation_predictions, label="forecast prediction")
plt.legend()


#true prediction
validation_target = Y[-N//2:]
validation_predictions=[]

#last_train_input len T
last_x = X[-N//2] #1D array of len T
while len(validation_predictions) < len(validation_target):
    p = model.predict(np.expand_dims(last_x, axis=0))[0,0]
    validation_predictions.append(p)
    
    #update last_x
    last_x=np.roll(last_x, -1)
    last_x[-1] = p    

plt.figure()    
plt.plot(validation_target, label="forecast target")
plt.plot(validation_predictions, label="forecast prediction")
plt.title("true")

plt.legend()





