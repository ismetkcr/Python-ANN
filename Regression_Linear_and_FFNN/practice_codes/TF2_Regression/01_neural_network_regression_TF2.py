# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:56:55 2024

@author: ismt
"""

#Intro to Regression with Nueral NN
# predicting numerical variable based on other combination of variables even shorter

import tensorflow as tf


#create data 

import numpy as np
import matplotlib.pyplot as plt

# Create features
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Create labels
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

#plt.scatter(X, y)

y == X + 10 #relationship between independent and dependent variables

#ınput and output shapes

#create a demo tensor for our housing price prediction

house_info = tf.constant(["bedroom", "bathroom", "garage" ])
house_price = tf.constant([939700])
print(house_info, house_price)

input_shape = X.shape
output_shape = y.shape
print(input_shape, output_shape)# 8, 8,

input_shape = X[0].shape
output_shape = y[0].shape
print(input_shape, output_shape)# 0, 0,

#turn numpy as tp tensor
X = tf.cast(tf.constant(X), dtype=tf.float32)
y = tf.cast(tf.constant(y), dtype=tf.float32)
input_shape = X[0].shape
output_shape = y[0].shape
print(input_shape, output_shape)# 0, 0,

#steps in modelling wtih TF
#1. Creating a model - define the input and output layers,
#as well as the hidden layers of deep learning model
#2. Compiling the model - define loss function and optimizer
#and evaluation metrics
#3. fit the model #gradient tape .. or just use fit

#set random seed
tf.random.set_seed(42)

# #create model another way
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(1))
# #same thing above


#create model ussing seq. API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
    ])

#compile model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics = ["mae"])

#fit the model
model.fit(tf.expand_dims(X, axis=-1), y, epochs=5)

#predict
y_pred = model.predict([17.0])

#we can imporove our model, by altering the steps we took to create a model..

#creating model -- add more layers,
#                  increase hidden units,
#                  change activation func

#compiling model -- change optimization func,
#                   learning_rate

#fitting model -- more epochs,
#                 more data


## ------- ## improved model..

#rebuild improved model


# Set random seed
tf.random.set_seed(42)

# Create a model (same as above)
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Compile model (same as above)
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

# Fit model (this time we'll train for longer)
model.fit(tf.expand_dims(X, axis=-1), y, epochs=100) # train for 100 epochs not 10
model.predict([17.0])


#another change to improve model

#model with extra hidden layer
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1)
    ])

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])

model.fit(tf.expand_dims(X, axis=-1), y, epochs=100)
model.predict([17.0]) #overfitting...


#evaluation a model
"""
when it comes to evaluation 3 words we need to remember
visualize, visualize, visualize..
The data - what data are we working with?
The model itself - what does our model look like?
the predictions of the model - how do the predictions of a model line up against the ground truth?
()
"""

# make bigger dataset
X = np.arange(-100, 100, 4)
#make labels for dataset
y = X + 10

#visualize data
plt.scatter(X, y)

### three sets...
#training set - the model learns from this data
# %70 80 
#validation set - the model gets tuned on this data
# %10 15
#test set
#the model gets evaluated on this data to test
# %10 15

# check the length of samples
print(len(X))

#split data train and test sets
X_train = X[:40] #first of 40 %80
y_train = y[:40]
 
X_test = X[40:] #last of 10 %20
y_test = y[40:]

print((len(X_train), len(X_test))) #40, 10
print((len(y_train), len(y_test))) #40, 10

### visualize data ###
#now we got train and test dataset visualize again...

plt.figure(figsize=(10,7))
#plot train data blue
plt.scatter(X_train, y_train, c="b", label="Training data")
plt.scatter(X_test, y_test, c="g", label="Testing data" )
plt.legend()


#lets look a make neural network..
#1 cerate layer
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
    ])

#compile
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])


#visualize model
model.summary()

#fit
#model.fit(tf.expand_dims(X_train, axis=-1), y, epochs=100)

#lets create a model which builds automatically by defining the input_shape argument in the first layer

tf.random.set_seed(42)
#create a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=[1], name="input_layer"),
    tf.keras.layers.Dense(1, name="output_layer")
    ], name="model_1")

#compile
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

model.summary()
#total params - total number of parameters in the model
#trainable parameters - these are the parameters (patterns)
#non trainable parameters - these parameters non updated during training #transfer learning..
model.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100, verbose=0)

from tensorflow.keras.utils import plot_model
plot_model(model=model) #gives error in spyder try notebook


#visualize predictions ...
#to visualize predictions plot them againg ground truth labels..
#y_true versus y_pred
#make some predictions
y_pred = model.predict(X_test)

#create plotting function..
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=y_pred):
    """
    plot training data, test data and compares predictions to labels
    """
    plt.figure(figsize=(10,7))
    #plot training data blue
    plt.scatter(train_data, train_labels, c="b", label="Training Data")
    #plot testing data green
    plt.scatter(test_data, test_labels, c="g", label="testing data")
    #plot models predictions in red
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    #show the legend
    plt.legend()
    
plot_predictions(train_data=X_train,
                 train_labels=y_train,
                 test_data=X_test,
                 test_labels=y_test,
                 predictions=y_pred)
    
#evaluationg predictions with evaluation metrics


#depending on problem there will be different metrics..
#since we're working on regression two of main metrics:
    #mae
    #mse
    

model.evaluate(X_test, y_test)

#calculate the main abs error
mae = tf.metrics.mean_absolute_error(y_true=y_test, y_pred=tf.squeeze(y_pred))

#calculate mean square error
mse = tf.metrics.mean_squared_error(y_true=y_test, y_pred=tf.squeeze(y_pred))

#make some functions to reuse mae, mse

def mae(y_true, y_pred):
    return tf.metrics.mean_absolute_error(y_true=y_true,
                                          y_pred=tf.squeeze(y_pred))

def mse(y_true, y_pred):
    return tf.metrics.mean_squared_error(y_true=y_true,
                                          y_pred=tf.squeeze(y_pred))

#running experiments to improve our model
#1 - get more data
#2 - make model larger(complex models, more layer, more hidden)
#3 train for longer..

#3 modelling experiments..
#1. model_1 same as original moderi 1 layer 100 epochs
#2. model_2 2 layers, 100 epochs
#3 model_3 2 layers trained 500 epochs
## build model_1##
tf.random.set_seed(42)

model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
    ])

model_1.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["mae"])

model_1.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100)
#make and plot predictions for model 1
y_preds_1 = model_1.predict(X_test)
#model_1 evaluate
mae_1 = mae(y_test, y_preds_1)
mse_1 = mse(y_test, y_preds_1)

tf.random.set_seed(42)
model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1)
    ])

model_2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["mse"])

model_2.fit(np.expand_dims(X_train, axis=-1), y_train, epochs=100)
y_preds_2 = model_2.predict(X_test)
mae_2 = mae(y_test, y_preds_2)
mse_2 = mse(y_test, y_preds_2)

#commpare with pandas
import pandas as pd
model_results = [
    ["model_1", mae_1, mse_1],
    ["model_2", mae_2, mse_2]
    ]

all_results = pd.DataFrame(model_results, columns=["model", "mae", "mse"])

# saving model..
""" tow main formatsss.
# SavedModel format
# the HDF5 format

"""

#save model
model_2.save("best_model_SavedModel_format") 
model_2.save("best_model_HDF5_format.h5") #its easier actually.. use it

#load model..
loaded_savedModel_format = tf.keras.models.load_model("/../..best_model_SavedModel_format") #give as path
loaded_h5_model = tf.keras.models.load_model("best_model_HDF5_format.h5")

