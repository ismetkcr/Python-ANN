# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 12:26:42 2025

@author: ismt
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Flatten, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#LOAD DATA
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test/255.0
print(f"x_train.shape = {x_train.shape}")

#build model
i = Input(shape=(x_train[0].shape))
x = LSTM(128)(i)
x = Dense(10, activation="softmax")(x)
model = Model(i, x)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=Adam(),
              metrics=["accuracy"])

hist = model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          epochs=10)


#plots.. accuracy and loss.

#show some missclasified exp
p_test = model.predict(x_test).argmax(axis=1)
mis_idx = np.where(p_test != y_test)[0]
i = np.random.choice(mis_idx)
plt.imshow(x_test[i], cmap="gray")
plt.show()
