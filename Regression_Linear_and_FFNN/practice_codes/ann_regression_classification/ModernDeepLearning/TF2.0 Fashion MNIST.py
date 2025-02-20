# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 19:01:25 2024

@author: ismt
"""
#libraries
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import numpy as np

#Load in the data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("before expand x.shapedims = " ,x_train.shape)

#the data is 2D, Conv expects HxWxC
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("after expand x.shapedims = " ,x_train.shape)

K = len(set(y_train)) #set for dissociate unique values
print("Number of classes = ", K)

#Build a model using fuctional API
i = Input(shape = x_train[0].shape)
x = Conv2D(32, (3,3), strides = 2, activation = 'relu')(i)
x = Conv2D(64, (3,3), strides = 2, activation = 'relu')(x)
x = Conv2D(128, (3,3), strides = 2, activation = 'relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(512, activation = 'softmax')(x)
model = Model(i, x)

#compile and fit
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
r = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 15)

#plot loss per iteration
plt.figure()
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()

#plot accuracy per iteration
plt.figure()
plt.plot(r.history['accuracy'], label = 'accuracy')
plt.plot(r.history['val_accuracy'], label = 'accuracy')
plt.legend()




