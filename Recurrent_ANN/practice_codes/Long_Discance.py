# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 21:50:09 2025

@author: ismt
"""

from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Flatten, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#build dataset
#this is nonlinear and long-distance dataset
#(actually we will test long distance vs short distance patterns)
#start with small T and increase it later

T = 10
D = 1
X = []
Y = []


def get_label(x, i1, i2, i3):
    #x = sequence
    if x[i1] < 0 and x[i2] < 0 and x[i3]<0:
        return 1
    if x[i1] < 0 and x[i2]>0 and x[i3]>0:
        return 1
    if x[i1] > 0 and x[i2] <0 and x[i3]>0:
        return 0
    if x[i1] > 0 and x[i2] >0 and x[i3] <0:
        return 1
    return 0

for t in range(5000):
    x=np.random.randn(T)
    X.append(x)
    #y=get_label(x, -1, -2, -3) #short distance
    y = get_label(x, 0, 1, 2) #long distance
    Y.append(y)
    
X=np.array(X)
Y=np.array(Y)
N = len(X)

#try a linear model first its now classification problem
i = Input(shape=(T,))
x = Dense(1, activation="sigmoid")(i)
model = Model(i, x)
model.compile(loss="binary_crossentropy",
              optimizer=Adam(learning_rate=0.01),
              metrics=["accuracy"])

hist = model.fit(X, Y,
                 epochs=100,
                 validation_split=0.5)


#plot loss
plt.figure()
plt.plot(hist.history["loss"], label="loss")
plt.plot(hist.history["val_loss"], label="val_loss")
plt.legend()
#plot accuracy
plt.figure()
plt.plot(hist.history["accuracy"], label="accuracy")
plt.plot(hist.history["val_accuracy"], label="val_accuracy")
plt.legend()

#Now try simple RNN
inputs = np.expand_dims(X, -1)
i = Input(shape=(T, D))

#method1
# 
#x = LSTM(5)(i)
# x = GRU(5)(i)
#x = SimpleRNN(5)(i)

#method2
x = LSTM(5, return_sequences=True)(i)
x = GlobalMaxPooling1D()(x)


x = Dense(1, activation="sigmoid")(x)
model = Model(i, x)
model.compile(loss="binary_crossentropy",
              optimizer=Adam(learning_rate=0.001),
              metrics=["accuracy"])

hist = model.fit(inputs, Y,
                 batch_size=1,
                 epochs=200,
                 validation_split=0.5)

#plot loss
plt.figure()
plt.plot(hist.history["loss"], label="loss")
plt.plot(hist.history["val_loss"], label="val_loss")
plt.legend()
#plot accuracy
plt.figure()
plt.plot(hist.history["accuracy"], label="accuracy")
plt.plot(hist.history["val_accuracy"], label="val_accuracy")
plt.legend()




    