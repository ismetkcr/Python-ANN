# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 22:44:57 2024

@author: ismt
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#read in the insurance dataset..

insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/insurance.csv")
#insurance_2 = pd.read_csv("insurance.csv")

print(insurance.head())
#print(insurance_2.head())


# insurance_one_hot = pd.get_dummies(insurance) #one hot encode
# failed to convert a NumPy array to a Tensor (Unsupported object type int).
insurance_one_hot = pd.get_dummies(insurance, dtype=int) #one hot encode we need this to convert trues and falses ints

#create X, and Y values (features and values)
X = insurance_one_hot.drop("charges", axis=1)
y = insurance_one_hot["charges"]

X.head()
y.head()

#create training and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

len(X), len(X_train), len(X_test)

#build neural network
tf.random.set_seed(42)

insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
    ])

insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.SGD(),
                        metrics=["mae"])

insurance_model.fit((X_train), y_train, epochs=100)

#insurance_model.fit(np.expand_dims(X_train, axis=-1), y_train, epochs=100)
#we dont need to use np.expand dims since xtrain already 2 dimension


#check the result of model on test data
insurance_model.evaluate(X_test, y_test)
y_train.median(), y_train.mean()

#models performance is bad, try to improve it

#
tf.random.set_seed(42)

insurance_model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(100), #too complex return nan for loss and mae..
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
    ])

insurance_model_2.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=["mae"])

#using adam rather than sgd error disappear in complex model

history = insurance_model_2.fit(X_train, y_train, epochs=100)
#history is useful evaluate

pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()






















