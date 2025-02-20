# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:52:30 2025

@author: ismt
"""

import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import LSTM, Embedding, TextVectorization
from tensorflow.keras.models import Model


df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df.head()

df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df.head()

df.columns = ['labels', 'data']
df.head()


#we need binary labels

df['binary_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
Y = df['binary_labels'].values

df_train, df_test, Ytrain, Y_test = train_test_split(df['data'], Y, test_size=0.33)


#create tf dataset
train_ds = tf.data.Dataset.from_tensor_slices((df_train.values, Ytrain))
test_ds = tf.data.Dataset.from_tensor_slices((df_test.values, Y_test))


#convert sentences to sequences
max_vocab_size=20_000
vectorization = TextVectorization(max_tokens=max_vocab_size)
vectorization.adapt(train_ds.map(lambda x, y: x))

#shuffle and batch as usual
train_ds = train_ds.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)

vectorization.get_vocabulary()

V = len(vectorization.get_vocabulary()) #kelime sayısı

#we get  embedded dimentionality hyperparameter
D = 20

#hidden state dim
M = 15


#this is inefficient, when we use network it takes strings one by one and vectorization every time

i = Input(shape=(1,), dtype=tf.string)
x = vectorization(i)
x = Embedding(V, D)(x) #matrix of size V, D
x = LSTM(M, return_sequences=True)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(i, x)

#compile and fit
model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

hist = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10
    )


#plot
plt.figure()
plt.plot(hist.history['loss'], label="loss")
plt.plot(hist.history['val_loss'], label="val_loss")
plt.legend()
plt.show()

#plot
plt.figure()
plt.plot(hist.history['accuracy'], label="accuracy")
plt.plot(hist.history['val_accuracy'], label="val_accuracy")
plt.legend()
plt.show()


df['labels'].hist()


from sklearn.metrics import f1_score
f1_score(Ytrain, model.predict(df_train.values) > 0.5)


#efficient way 
train_ints = vectorization(df_train.values)
test_ints = vectorization(df_test.values)
#create tf dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_ints, Ytrain))
test_ds = tf.data.Dataset.from_tensor_slices((test_ints, Y_test))


#shuffle and batch as usual
train_ds = train_ds.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)



#we get  embedded dimentionality hyperparameter
D = 20

#hidden state dim
M = 15


#this is inefficient, when we use network it takes strings one by one and vectorization every time

i = Input(shape=(None,))
x = Embedding(V, D)(i) #matrix of size V, D
x = LSTM(M, return_sequences=True)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(i, x)

#compile and fit
model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

hist = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10
    )



