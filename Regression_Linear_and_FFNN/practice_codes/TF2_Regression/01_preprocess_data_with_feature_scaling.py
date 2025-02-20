# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:02:10 2024

@author: ismt
"""
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#NORMALIZATION and STANDARDIZATION
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/insurance.csv")
#insurance_2 = pd.read_csv("insurance.csv")

print(insurance.head())
#print(insurance_2.head())

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

#create column transformer

ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]),
    #turn all values in these columns between 0, 1
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
    )
    


#create X and y values
X = insurance.drop("charges", axis=1)
y = insurance["charges"]

X_train, X_test, y_train, y_test = train_test_split(X,
                y, test_size=0.2, random_state=42)

#fit column transformer

ct.fit(X_train)

X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

print(X_train_normal[0])

X_train.shape, X_train_normal.shape
#we added extra columns one hot encode transform layer


tf.random.set_seed(42)

insurance_model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
    ])

insurance_model_2.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=["mae"])

insurance_model_2.fit(X_train_normal, y_train, epochs=100)

#Evaluate normalized data
insurance_model_2.evaluate(X_test_normal, y_test)





















