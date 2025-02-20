# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:06:02 2025

@author: ismt
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#things i have to  know
# N = number of samples (number of sequences)
# T = sequence length
# D = number of input features
# M = number of hidden units
# K = number of output units

#MAKE some DATA
N = 1
T = 10
D = 3
K = 2
X = np.random.randn(N, T, D)

#MAKE RNN
M = 5 #number of hidden layers
i = Input(shape=(T, D))
x = SimpleRNN(M)(i)
x = Dense(K)(x)
model=Model(i, x)

Yhat = model.predict(X)


#check shapes
#first output is input to hidden
#second output is hidden to hidden
#third output is bias term (vector of len N)

a, b, c = model.layers[1].get_weights()
print(a.shape, b.shape, c.shape)

Wx, Wh, bh = model.layers[1].get_weights()
Wo, bo = model.layers[2].get_weights()

h_last = np.zeros(M) #initial hidden state
x = X[0] #the one and only sample
Yhats=[]

for t in range(T):
    h = np.tanh(x[t].dot(Wx) + h_last.dot(Wh) + bh)
    y = h.dot(Wo) + bo
    Yhats.append(y)
    
    h_last = h

print(Yhats[-1])



