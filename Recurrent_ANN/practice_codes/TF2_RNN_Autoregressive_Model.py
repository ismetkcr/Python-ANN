# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:38:11 2025

@author: ismt
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#make original data
series = np.sin(0.1*np.arange(200)) + np.random.randn(200)*0.1
plt.plot(series)
plt.show()


#build dataset
#lets see if  we can use T past values to predict next value
T = 10
X = []
Y = []
for t in range(len(series) - T):
    x=series[t:t+T]
    X.append(x)
    y=series[t+T]
    Y.append(y)
    
X = np.array(X).reshape(-1, T)
Y = np.array(Y)
N = len(X)
print(f"X.shape is {X.shape}\n Y.shape is {Y.shape}, N={N}")

#try build autoregressive linear model
inputs = Input(shape=(T,), name="input_layer")
outputs = Dense(1, name= "output_layer")(inputs)
model = Model(inputs, outputs)
model.compile(loss="mse",
              optimizer=Adam(lr=0.1))

#train RNN
history_1 = model.fit(
    X[:-N//2], Y[:-N//2],
    batch_size=1,
    epochs=80,
    validation_data=(X[-N//2:],  Y[-N//2:]),
    )

#plot loss
loss = history_1.history["loss"]
val_loss = history_1.history['val_loss']

df = pd.DataFrame({
    "loss": loss,
    "val_loss": val_loss
     })


plt.figure()
plt.plot(df["loss"], label="loss", c="r")
plt.plot(df["val_loss"], label="val_loss", c="g" )
plt.xlabel("epochs")
plt.ylabel("loss")
plt.plot()


#wrong forecast using true targets
#we are using true X's in our predictions
validation_target = Y[-N//2:]
validation_predictions = []

#index of first validation_input
i = -N//2
while len(validation_predictions) < len(validation_target):
    p = model.predict(X[i].reshape(1, -1))[0, 0] #1x1 array scalar #same as np.expand_dims(X[i], axis=0)
    i+=1
    
    validation_predictions.append(p)

plt.figure()    
plt.plot(validation_target, label="forecast target")
plt.plot(validation_predictions, label="forecast prediction")
plt.legend()
plt.show()

#true way to make predictions
validation_target = Y[-N//2:]
validation_predictions = []
last_x = X[-N//2] #last train data #1D array of X
while len(validation_predictions) < len(validation_target):
    p = model.predict(last_x.reshape(1, -1))[0,0]
    validation_predictions.append(p)
    
    #construct new last_x
    last_x = np.roll(last_x, -1)
    last_x[-1] = p
    
plt.figure()    
plt.plot(validation_target, label="forecast target")
plt.plot(validation_predictions, label="forecast prediction")
plt.legend()
plt.show()

model.evaluate(X[-N//2:],Y[-N//2:] )