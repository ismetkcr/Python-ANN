# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 19:59:13 2025

@author: ismt
"""

import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization, Input, Dense, Embedding
from tensorflow.keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model

df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df.head()

df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df.columns = ['labels', 'data']
df.head()

df['binary_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
Y = df['binary_labels'].values

df_train, df_test, Ytrain, Ytest = train_test_split(df['data'], Y, test_size=.33)

#create tf datasets
train_ds = tf.data.Dataset.from_tensor_slices((df_train.values, Ytrain))
test_ds = tf.data.Dataset.from_tensor_slices((df_test.values, Ytest))


#sentence to sequence
max_vocab_size=20_000
vectorization = TextVectorization(max_tokens = max_vocab_size)
vectorization.adapt(train_ds.map(lambda x, y:x)) #layer only see data.. dont label (train_ds contains labels also)

train_ds = train_ds.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)

V = len(vectorization.get_vocabulary())

input_sequences_train = vectorization(df_train.values)
input_sequences_test = vectorization(df_test.values)

input_sequences_train.shape
input_sequences_test.shape


T = input_sequences_train.shape[1]


vectorization2 = TextVectorization(
    max_tokens=max_vocab_size,
    output_sequence_length=T,
    vocabulary=vectorization.get_vocabulary()
    )



input_sequences_train = vectorization2(df_train.values)
input_sequences_test = vectorization2(df_test.values)



#create model
D = 20
i = Input(shape=(1,), dtype=tf.string)
x = vectorization2(i)
x = Embedding(V, D)(x)
x = Conv1D(32, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(64, 3, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(i, x)

model.compile(
    loss = "binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
    )

hist = model.fit(train_ds,
          validation_data=test_ds,
          epochs=5)



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

#another way to train..
#create tf datasets
train_ds = tf.data.Dataset.from_tensor_slices((input_sequences_train, Ytrain))
test_ds = tf.data.Dataset.from_tensor_slices((input_sequences_test, Ytest))


train_ds = train_ds.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)


#create model
D = 20
i = Input(shape=(T,))
x = Embedding(V, D)(i)
x = Conv1D(32, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(64, 3, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(i, x)
model.compile(
    loss = "binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
    )

hist = model.fit(train_ds,
          validation_data=test_ds,
          epochs=5)



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
