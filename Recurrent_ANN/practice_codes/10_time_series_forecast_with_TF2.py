# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:11:12 2025

@author: ismt
"""

#fundemantels .. + project bitpredict
#NOTE: This is only for learning purposes not for financial advice 

#get data 
#going to be use historical price data of bitcoin to try an predict the future price of data
#import time series data with pandas its best i think 

import pandas as pd
#read bitcoin data and parse dates
df = pd.read_csv("BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv",
                 parse_dates=["Date"],
                 index_col=["Date"]) #parse date column and tell pandas column 1 is ad datetime

df.head()

df.info()

#how many samples do we have?
len(df)
#We've colledted historical price of btc of past 8 years but we have 2787 samples
#ml models like lots of data.

#closing price for each day
bitcoin_prices = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)" : "Price"})

import matplotlib.pyplot as plt

bitcoin_prices.plot()
plt.ylabel("btc price")
plt.title("Price of BTC from 1 oct 2013 to 18 may 2021")
plt.legend()

#read file with python csv

##import time series data with PYTHON csv module
import csv
from datetime import datetime

timesteps=[]
btc_price=[]

with open("BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv", "r") as f:
    csv_reader=csv.reader(f, delimiter=",")
    next(csv_reader) #its going to skip the header (first line is column titles)
    for line in csv_reader:
        #timesteps.append(line[1])
        timesteps.append(datetime.strptime(line[1], "%Y-%m-%d")) #string parse time
        btc_price.append(float(line[2])) #get closing price as float

#view forst 10
timesteps[:10], btc_price[:10]

#plot from csv
import numpy as np
plt.figure()
plt.plot(timesteps, btc_price, label="btcprice")
plt.title("Price of BTC from 1 oct 2013 to 18 may 2021")
plt.ylabel("btc price")
plt.xlabel("date")
plt.legend()

#format data part 1 : create train and test split time series data
#!!!!!!!! WRONG WAY !!!!!!!!!!!!!!!!
#using sklearn train test split
#get btc data array 
timesteps = bitcoin_prices.index.to_numpy()
prices = bitcoin_prices.Price.to_numpy()
timesteps[:10], prices[:10]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(timesteps, #dates
                                                    prices, #btc price
                                                    test_size=0.2,
                                                    random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
#plot wrong train and test splits
plt.figure()
plt.scatter(X_train, y_train, s=5, label="train_data")
plt.scatter(X_test, y_test, s=5, label="test_data")
plt.xlabel("date")
plt.ylabel("pricee")
plt.legend()
plt.show()

#create train and test sets for time series right way
split_size=int(0.8 * len(prices)) # %80 train %20 test

#create train data 
X_train, y_train = timesteps[:split_size], prices[:split_size]
X_test, y_test = timesteps[split_size:], prices[split_size:]

len(X_train), len(X_test), len(y_train), len(y_test)
#plot true train and test splits
plt.figure()
plt.scatter(X_train, y_train, s=5, label="train_data")
plt.scatter(X_test, y_test, s=5, label="test_data")
plt.xlabel("date")
plt.ylabel("pricee")
plt.legend()
plt.show()

#Create plotting function 

def plot_time_series(timesteps, values, format=".", start=0, end=None, label="None"):
    #plot series
    #plt.figure()
    plt.plot(timesteps[start:end],values[start:end], format, label=label)
    plt.xlabel("time")
    plt.ylabel("btc price")
    if label:
        plt.legend()
    plt.grid(True)

#test function
plot_time_series(timesteps=X_train, values=y_train, label="Train_data")
plot_time_series(timesteps=X_test, values=y_test, label="test_Data")

#modelling experiments..
"""
0 : Naive
1 : dense with horizon 1 window = 7
2 : dense with horizon 1 window = 30
3 : dense with horizon 7 window = 30
4 : conv1d model
5 : LSTM
6 : same as 1 with multivariate data
7 : N-Beats Algorithm
8 : Ensemble Multiple Models
9 : Future prediction model
10 : same asmodel 1 with turkey

"""

#model 0 - naive forecast baseline..
#the formula looks like this
#y(t)_hat = y(t-1)
#the predictioon at time step t is equal to value of t-1
y_test[:10]
naive_forecast = y_test[:-1]

#plot naive forecast
plt.figure()
#plot_time_series(timesteps=X_train, values=y_train, label="train data")
plot_time_series(timesteps=X_test, values=y_test, start=350, label="test_Data")
plot_time_series(timesteps=X_test[1:], values=naive_forecast, start=350, format="-", label="predicted naive value")

#evaluation metrics
# MAE, MSE, 
#how do out models forecast compared the actual values...
#Implement MASE ın TF mean absolute scaled error
import tensorflow as tf
def mean_absolute_scaled_error(y_true, y_pred):
    """
    Implement MASE
    Formula: q(j) = e(j) / (1/(T-1) * sum(|y_t - y_t-1|))
    # T = len(y_true)
    # if T!= len(y_pred):
    #     raise ValueError("length..")
    # q = []
    # for j in range(T):
    #     e_j = abs(y_true[j] - y_pred[j])
    #     int_j = 0
    #     for t in range(1,T):
    #         int_j += abs(y_true[t] - y_true[t-1])
    #     scaling_factor = int_j / (T -1)
    #     if scaling_factor == 0:
    #         raise ValueError(f"zero at point{j}")
    #     q_j = e_j/scaling_factor
    #     q.append(q_j)
    # return q


    Parameters
    ----------
    y_true : TYPE
        DESCRIPTION.
    y_pred : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    #find mae for naive forecast seaosinality
    mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1])) #seasionality day is 1 
    return mae / mae_naive_no_season
    
    
    # #naive 
    
mean_absolute_scaled_error(y_true=y_test[1:],
                           y_pred=naive_forecast).numpy()


def evaluate_preds(y_true, y_pred):
    #make sure datatype
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    #metrics
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)
    
    return {"mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy(),
            "mase": mase.numpy()}

naive_results = evaluate_preds(y_true = y_test[1:],
                               y_pred = naive_forecast)

#use for baselines and actual forecast
"""
MovingAverage..
ARIMA
sktime
...

"""
    
#Windowing our dataset

"""
Horizon : number of timesteps we are going to predict in future
Window : number of timesteps we are using to predict in past
Window for one week
[0, 1, 2, 3, 4, 5, 6] -->[7]
[1, 2, 3, 4, 5, 6, 7] -->[8]
[2, 3, 4, 5, 6, 7, 8] -- [9]

"""

print(f"We want to use {btc_price[:7]} to predict this: {btc_price[7]}")
HORIZON=1 #predict next one day
WINDOW_SIZE=7 #use the past week 

#create function to label windowed data
btc_price[:10]
# Create function to label windowed data
def get_labelled_windows(x, horizon=1):
  """
  Creates labels for windowed dataset.

  E.g. if horizon=1 (default)
  Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
  """
  return x[:, :-horizon], x[:, -horizon:]

"""
x : array([1, 2, 3, 4, 5, 6, 7, 8])
x[:-1] --> 1,2,3,4,5,6,7
x[-1:] --> 8



"""
    
#test window label function
test_window, test_label = get_labelled_windows(tf.expand_dims(tf.range(8), axis=0))
test_window, test_label
# N = len(train_data)
# fot t in range(N-WINDOW_SIZE):
#     window, label = get_labelled_window[t:t+WINDOW_SIZE]

#we could to this with for loop but for speed 
#numpy array indexing

#1 create window step of specific window size
#2 use numpy indexing to create 2D array of multiplewindowsteps
#3 use 2D array of multiple window steps from 2 to index on a target series
#4 use get_labelled_windows

# Create function to view NumPy arrays as windows 
def make_windows(x, window_size=7, horizon=1):
  """
  Turns a 1D array into a 2D array of sequential windows of window_size.
  """
  # 1. Create a window of specific window_size (add the horizon on the end for later labelling)
  window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
  # print(f"Window step:\n {window_step}")

  # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
  window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T # create 2D array of windows of size window_size
  # print(f"Window indexes:\n {window_indexes[:3], window_indexes[-3:], window_indexes.shape}")

  # 3. Index on the target array (time series) with 2D array of multiple window steps
  windowed_array = x[window_indexes]

  # 4. Get the labelled windows
  windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

  return windows, labels
    
    
    
full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
len(full_windows), len(full_labels)

for i in range(3):
    print(f"window: {full_windows[i]} -- > Label : {full_labels[i]}")
    

for i in range(3):
    print(f"window : {full_windows[i-3]} -- > label : {full_labels[i-3]}")


#Also we can use easily tf.keras.preprocessing.timeseries_dataset_from_array

# Make the train/test splits
def make_train_test_splits(windows, labels, test_split=0.2):
  """
  Splits matching pairs of windows and labels into train and test splits.
  """
  split_size = int(len(windows) * (1-test_split)) # this will default to 80% train/20% test
  train_windows = windows[:split_size]
  train_labels = labels[:split_size]
  test_windows = windows[split_size:]
  test_labels = labels[split_size:]
  return train_windows, test_windows, train_labels, test_labels


train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)
len(train_windows), len(test_windows), len(train_labels), len(test_labels)

import os

# Create a function to implement a ModelCheckpoint callback with a specific filename 
def create_model_checkpoint(model_name, save_path="model_experiments"):
  return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name), # create filepath to save model
                                            verbose=0, # only output a limited amount of text
                                            save_best_only=True) # save only the best model to file
    

#MODEL 1 : Dense model (window=7, horizon=1)
from tensorflow.keras import layers

tf.random.set_seed(42)

#constructmodel
model_1 = tf.keras.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(HORIZON, activation="linear") #same as none
    ], name="model_1_dense")

model_1.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["mae", "mse"])

model_1.fit(x=train_windows,
            y=train_labels,
            epochs=100,
            verbose=1,
            batch_size=128,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_name=model_1.name)])


#evaluate model on test data
model_1.evaluate(test_windows, test_labels)

#load saved best performing model 
model_1 = tf.keras.models.load_model("model_experiments/model_1_dense/")
model_1.evaluate(test_windows, test_labels)


#forecast test dataset (pseudo)
#function to take model, data returns predictions
def make_preds(model, input_data):
  """
  Uses model to make predictions on input_data.

  Parameters
  ----------
  model: trained model 
  input_data: windowed input data (same kind of data model was trained on)

  Returns model predictions on input_data.
  """
  forecast = model.predict(input_data)
  return tf.squeeze(forecast) # return 1D array of predictions

model_1_preds = make_preds(model_1, test_windows)
len(model_1_preds), model_1_preds[:10]

model_1_results = evaluate_preds(y_true=tf.squeeze(test_labels),
                                 y_pred=model_1_preds)

model_1_results

#plot predictions
offset = 300
plt.figure()
plot_time_series(timesteps=X_test[-len(test_windows):], 
                 values=test_labels[:,0],
                 start=offset, 
                 label="test_data")

plot_time_series(timesteps=X_test[-len(test_windows):],
                 values=model_1_preds,
                 start=offset,
                 format="-",
                 label="model_1_preds")



#Create same model as 1 but window size=30, horizon=1
#MODEL 2 ..
WINDOW_SIZE=30 #30 day past
HORIZON = 1
#make windowed data with new window size
full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
len(full_windows), len(full_labels)
train_windows, test_windows, train_labels, test_labels = make_train_test_splits(windows=full_windows,
                                                                                labels=full_labels,
                                                                                test_split=0.2)
len(train_windows), len(test_windows), len(train_labels), len(test_labels)

tf.random.set_seed(42)
model_2 = tf.keras.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(HORIZON),
    ], name="model_2_dense")

model_2.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam())

model_2.fit(train_windows,
            train_labels,
            epochs=100,
            batch_size=128,
            verbose=0,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_name=model_2.name)])

#evaluate 
model_2.evaluate(test_windows, test_labels)

model_2 = tf.keras.models.load_model("model_experiments/model_2_dense/")
model_2.evaluate(test_windows, test_labels)


model_2_preds = make_preds(model_2,
                           input_data=test_windows)

model_2_results = evaluate_preds(y_true=tf.squeeze(test_labels),
                                 y_pred=model_2_preds)

offset = 300
plt.figure()
plot_time_series(timesteps=X_test[-len(test_windows):],
                 values=test_labels[:, 0],
                 start=offset,
                 label="test_data")

plot_time_series(timesteps=X_test[-len(test_windows):],
                 values=model_2_preds,
                 start=offset,
                 format="-",
                 label="model_2_preds")


#experiment 3: same model as 1, but window=30, horizon=7
#MODEL 3  (same as 1)

HORIZON=7
WINDOW_SIZE=30
full_windows_new, full_labels_new = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
len(full_windows_new), len(full_labels_new)
full_windows = full_windows_new
full_labels = full_labels_new
len(full_windows), len(full_labels)

train_windows_new, test_windows_new, train_labels_new, test_labels_new = make_train_test_splits(windows=full_windows,
                                                                                                labels=full_labels)
train_windows, test_windows, train_labels, test_labels = train_windows_new, test_windows_new, train_labels_new, test_labels_new


len(train_windows), len(test_windows), len(train_labels), len(test_labels)

tf.random.set_seed(42)

model_3 = tf.keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(HORIZON)
    ], name="model_3_dense")

model_3.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam())

model_3.fit(train_windows,
            train_labels,
            batch_size=128,
            epochs=100,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_name=model_3.name)]
            )

model_3.evaluate(test_windows, test_labels)

model_3 = tf.keras.models.load_model("model_experiments/model_3_dense/")
model_3.evaluate(test_windows, test_labels)

model_3_preds = make_preds(model_3,
                           input_data=test_windows)

model_3_results = evaluate_preds(y_true=tf.squeeze(test_labels),
                                 y_pred=model_3_preds)
                                                                                            
#correct evaluate_preds function for dealing higher shapes
def evaluate_preds(y_true, y_pred):
    #make sure datatype
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    #metrics
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)
    
    if mae.ndim>0:
        mae=tf.reduce_mean(mae)
        mse=tf.reduce_mean(mse)
        rmse=tf.reduce_mean(rmse)
        mape=tf.reduce_mean(mape)
        mase=tf.reduce_mean(mase)
    
    
    
    
    return {"mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy(),
            "mase": mase.numpy()}


model_3_results = evaluate_preds(y_true=tf.squeeze(test_labels),
                                 y_pred=model_3_preds)

model_3_results

#visualize predictions
offset = 300
plt.figure()
plot_time_series(timesteps=X_test[-len(test_windows):],
                 values=test_labels[:,0],
                 start=offset,
                
                 label="test data")

plot_time_series(timesteps=X_test[-len(test_windows):],
                 values=tf.reduce_mean(model_3_preds, axis=1),
                 start=offset,
                 format="-",
                 label="model_3_preds")


#best model so far?
pd.DataFrame({"naive": naive_results["mae"],
              "horizon_1_window_7": model_1_results["mae"],
              "horizon_1_window_30": model_2_results["mae"],
              "horizon_7_window_30": model_3_results["mae"]
              }, index=["mae"]).plot(kind="bar")


#Model 4Conv1D
HORIZON = 1
WINDOW=SIZE=7

full_windows_new2, full_labels_new2 = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
len(full_windows_new2), len(full_labels_new2)

train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows_new2, full_labels_new2)
len(train_windows), len(test_windows), len(train_labels), len(test_labels)


#To use conv1d layer input shape shuold be (batch_size, timesteps, input_dim)
#check data input shape
train_windows[0].shape #returns window_size
#before we pass our data to conv1d layar we have to reshape it to make sure it works
x = tf.constant(train_windows[0])
expand_dims_layer = layers.Lambda(lambda x: tf.expand_dims(x, axis=1)) #add extra dimension for input dim

print(f"Original shape: {x.shape}")
print(f"expanded shape : {expand_dims_layer(x).shape}")
print(f"original values with expanded shape :\n {expand_dims_layer(x)}")



tf.random.set_seed(42)
model_4 = tf.keras.Sequential([
    layers.Lambda(lambda x: tf.expand_dims(x, axis=1)),
    layers.Conv1D(filters=128, kernel_size=5, strides=1, padding="causal", activation="relu"),
    layers.Dense(HORIZON)
    ], name="model_4_conv1D")


model_4.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam())

model_4.fit(train_windows,
            train_labels,
            batch_size=128,
            epochs=100,
            verbose=0,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_name=model_4.name)])

model_4.evaluate(test_windows, test_labels)

model_4 = tf.keras.models.load_model("model_experiments/model_4_conv1D")
model_4.evaluate(test_windows, test_labels)

model_4_preds = make_preds(model_4, test_windows)

model_4_results = evaluate_preds(y_true=tf.squeeze(test_labels),
                                 y_pred=model_4_preds)

#MODEL 5 LSTM
tf.random.set_seed(42)

inputs = layers.Input(shape=(WINDOW_SIZE))
x = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)
#x = layers.LSTM(128, return_sequences=True)(x)
x = layers.LSTM(128, activation="relu")(x)
#x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(HORIZON)(x)
model_5 = tf.keras.Model(inputs=inputs,
                         outputs=outputs,
                         name="model_5_LSTM")

model_5.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam())

model_5.fit(train_windows,
            train_labels,
            epochs=100,
            verbose=1,
            batch_size=128,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_name=model_5.name)])

model_5.evaluate(test_windows, test_labels)

model_5 = tf.keras.models.load_model("model_experiments/model_5_LSTM")
model_5.evaluate(test_windows, test_labels)

model_5_preds = make_preds(model_5, test_windows)
model_5_results = evaluate_preds(y_true=tf.squeeze(test_labels),
                                 y_pred=model_5_preds)


#Multivariate Model
#one feature i can add did i sleep well today... ?
#or i can add bitcoin halving events.. it would be much helpier maybe
bitcoin_prices.head()

block_reward_1 = 50 #3 january 2009 #this block rewards is not in our dataset it starts 2010
block_reward_2 = 25 # 8 november 2012
block_reward_3 = 12.5 # 9 july2016
block_reward_4 = 6.25 # 18 may 2020

#block reward dates
block_reward_2_datetime = np.datetime64("2012-11-28")
block_reward_3_datetime = np.datetime64("2016-07-09")
block_reward_4_datetime = np.datetime64("2020-05-18")

#create date ranges of where specific block_reward values should be
block_reward_2_days = (block_reward_3_datetime - bitcoin_prices.index[0]).days
block_reward_3_days = (block_reward_4_datetime - bitcoin_prices.index[0]).days
# add block reward values as a feature in dataframe bitcoin_prices
bitcoin_prices_block = bitcoin_prices.copy()
bitcoin_prices_block["block_reward"] = None
bitcoin_prices_block.head()

bitcoin_prices_block.iloc[:block_reward_2_days, -1] = block_reward_2
bitcoin_prices_block.iloc[block_reward_2_days:block_reward_3_days, -1] = block_reward_3
bitcoin_prices_block.iloc[block_reward_3_days:, -1] = block_reward_4

#plot block reward vs time
from sklearn.preprocessing import minmax_scale
scaled_price_block_df = pd.DataFrame(minmax_scale(bitcoin_prices_block[["Price", "block_reward"]]),
                                                                       columns=bitcoin_prices_block.columns,
                                                                       index=bitcoin_prices_block.index)
bitcoin_prices_block.plot()
scaled_price_block_df.plot()

#MultiVariateTimeSeries windowed dataset 
#previously we turned out univariate time series into windwow datset using function but now he have multivariate data
#we can use pandas.DataFrame.shift method 
#dataset hyperparams
HORIZON = 1 
WINDOW_SIZE = 7

bitcoin_prices_windowed = bitcoin_prices_block.copy()

#add windowed columns
for i in range(WINDOW_SIZE): #shift values for each step in window_size
    bitcoin_prices_windowed[f"Price+{i+1}"] = bitcoin_prices_windowed["Price"].shift(periods=i+1)

bitcoin_prices_windowed.head(10)
#create x windows and y horizons
X = bitcoin_prices_windowed.dropna().drop("Price", axis=1).astype(np.float32)
y = bitcoin_prices_windowed.dropna()["Price"].astype(np.float32)

# [0, 1, 2, 3, 4, 5, 6, block_reward] -> [7]
# [1, 2, 3, 4, 5, 6, 7, block_reward] -> [8]
# [2, 3, 4, 5, 6, 7, 8, block_reward] -> [9]

#make train and test set using indexing
split_size = int(len(X)*0.8)
X_train, y_train = X[:split_size], y[:split_size]
X_test, y_test = X[split_size:], y[split_size:]
len(X_train), len(X_test), len(y_train), len(y_test)


#Model 6: Dense (Multivariate time series)
model_6 = tf.keras.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(HORIZON)
    ], name="model_6_dense_multivariate")

model_6.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam())

model_6.fit(X_train, y_train,
            epochs=100,
            batch_size=128,
            verbose=1,
            validation_data=(X_test, y_test),
            callbacks=[create_model_checkpoint(model_name="model_6_dense_multivariate")])

model_6.evaluate(X_test, y_test)

model_6 = tf.keras.models.load_model("model_experiments\model_6_dense_multivariate")
model_6.evaluate(X_test, y_test)
                                    
#make predictions
model_6_preds = tf.squeeze(model_6.predict(X_test))
model_6_results = evaluate_preds(y_true=y_test,
                                 y_pred=model_6_preds)


#create NBeatBlock custom layer using tf subclassing
class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(self,input_size:int, theta_size:int,
                 horizon:int, n_neurons:int, n_layers:int,
                 **kwargs): #take cares of all the arguments from parent class
        super().__init__(**kwargs)
        self.input_size=input_size
        self.theta_size=theta_size
        self.horizon=horizon
        self.n_neurons=n_neurons
        self.n_layers=n_layers
        
        #block contains stack of 4 FC with rely
        self.hidden=[tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
        #output theta layer with linear activation
        self.theta_layer = tf.keras.layers.Dense(theta_size,activation="linear", name="theta")
        
    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x=layer(x)
        theta=self.theta_layer(x)
        #output
        backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
        return backcast, forecast
        
#test NBeat with dummy inputs
dummy_nbeats_block_layer = NBeatsBlock(input_size=WINDOW_SIZE,
                                      theta_size=WINDOW_SIZE+HORIZON,
                                      horizon=HORIZON,
                                      n_neurons=128,
                                      n_layers=4)    
        
    
tf.random.set_seed(42)
dummy_inputs = tf.expand_dims(tf.range(WINDOW_SIZE) +1, axis=0)#input shape 
dummy_inputs

#pass dummy inputs to nbeatch block layer
backcast, forecast = dummy_nbeats_block_layer(dummy_inputs)
print(f"Backcast : {tf.squeeze(backcast.numpy())}")
print(f"Forecast : {tf.squeeze(forecast.numpy())}")

#create data for NBEATS algorithm with tf.data
HORIZON = 1
WINDOW_SIZE = 7

#create n-beats data inputs univariate
bitcoin_prices.head()
bitcoin_prices_nbeats = bitcoin_prices.copy()
#X, y = make_windows(bitcoin_prices_nbeats["Price"].values, window_size=WINDOW_SIZE, horizon=HORIZON)
for i in range(WINDOW_SIZE):
  bitcoin_prices_nbeats[f"Price+{i+1}"] = bitcoin_prices_nbeats["Price"].shift(periods=i+1)
bitcoin_prices_nbeats.dropna().head()
# Make features and labels
X = bitcoin_prices_nbeats.dropna().drop("Price", axis=1)
y = bitcoin_prices_nbeats.dropna()["Price"]
split_size = int(len(X)*0.8)
X_train, y_train = X[:split_size], y[:split_size]
X_test, y_test = X[split_size:], y[split_size:]
len(X_train), len(y_train), len(X_test), len(y_test)


#make dataset tfdata
train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train)
train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)

#make dataset tfdata
test_features_dataset = tf.data.Dataset.from_tensor_slices(X_test)
test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test)

train_dataset = tf.data.Dataset.zip(train_features_dataset, train_labels_dataset)
test_dataset = tf.data.Dataset.zip(test_features_dataset, test_labels_dataset)

#batch and prefetch
BATCH_SIZE = 1024
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_dataset, test_dataset


#setting up NBEATS model hyperparams using paper
N_EPOCHS = 5000
N_NEURONS = 512
N_LAYERS = 4
N_STACKS = 30

INPUT_SIZE = WINDOW_SIZE * HORIZON
THETA_SIZE = INPUT_SIZE + HORIZON

INPUT_SIZE, THETA_SIZE

#get ready for residual connections
#2 need 2 layers for residual connections subtract and add
#Make tensors

tensor_1 = tf.range(10) + 10
tensor_2 = tf.range(10)

tensor_1, tensor_2, 

subtracted = tf.keras.layers.subtract([tensor_1, tensor_2])
added = layers.add([tensor_1, tensor_2])

print(subtracted)
print(added)


#Build Model
#1. Setup an instance of the N-BEATs block layer using NBeatsBlock
#(this will be the initial block used for the network, the rest will be created as part of stacks)

#2. create ınput layer for the N-Beats stack 
#3 make initial backcast and forecast for he model with layer created in 1
#4 use for loop for stack block layers
#5use nbeatcsblock class within the for loop to create blocks which return backcast and forecasts
#6 create double residual stacking using substract and add layers
#7 input and outputs together
#8 comoile with mae loss
#9 fit the model 

tf.random.set_seed(42)
#setup instance of n beats 
nbeats_block_layer = NBeatsBlock(input_size=INPUT_SIZE, 
                                 theta_size=THETA_SIZE,
                                 horizon=HORIZON, 
                                 n_neurons=N_NEURONS, 
                                 n_layers=N_LAYERS,
                                 name="InitialBlock")

#2. create input to stack
stack_input = layers.Input(shape=(INPUT_SIZE), name="stack_input")

#3create initial backcast and forecast 
backcast, forecast = nbeats_block_layer(stack_input)
residuals=layers.subtract([stack_input, backcast], name="subtract_00")


for i, _ in enumerate(range(N_STACKS-1)): #first stack already created in 3
    #5 use the nbeatsblock to calculate backcast and forecast
    backcast, block_forecast = NBeatsBlock(input_size=INPUT_SIZE,
                                           theta_size=THETA_SIZE,
                                           horizon=HORIZON,
                                           n_neurons=N_NEURONS,
                                           n_layers=N_LAYERS,
                                           name=f"NBeatsblock_{i}")(residuals)
    #6create double residual stacking
    residuals=layers.subtract([residuals,backcast], name=f"subtract_{i}")
    forecast=layers.add([forecast, block_forecast], name=f"add_{i}")
    
#7 put the stack model together
model_7 = tf.keras.Model(inputs=stack_input, outputs=forecast, name="model_7_NBEATS")

#8 compile
model_7.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam())

#9 fit the model
model_7.fit(train_dataset,
            epochs=N_EPOCHS,
            validation_data=test_dataset,
            verbose=1,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                        patience=200,
                                                        restore_best_weights=True),
                       tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                            patience=100,
                                                            verbose=1)])

model_7.evaluate(test_dataset)
model_7_preds = make_preds(model_7, test_dataset)
model_7_results = evaluate_preds(y_true=y_test,
                                 y_pred=model_7_preds)

#model 8 model ensemble
from tensorflow.keras import layers
#combines different models to predict common goal
def get_ensemble_models(horizon=HORIZON,
                        train_data=train_dataset,
                        test_data=test_dataset,
                        num_iter=10,
                        num_epochs=1000,
                        loss_fn=["mae", "mse", "mape"]):
    
    #turns a list of num_iter models each trained on MAE, MSE, MAPE loss
    #for example if num_iter = 10 and len(loss_fn)=3, them 30 trained models will be returned..
    #make empty list
    ensemble_models = []
    for i in range(num_iter):
        #build and fit new model with different loss fn
        for loss_function in loss_fn:
            print(f"Optimizing model by reducing: {loss_function} for {num_epochs} epochs, model_number{i}")
            #construct simple model
            model = tf.keras.Sequential([
                layers.Dense(128, kernel_initializer="he_normal", activation="relu"),
                layers.Dense(128, kernel_initializer="he_normal", activation="relu"),
                layers.Dense(HORIZON)
                ])
            
            #compile with current fn
            model.compile(loss=loss_function,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=["mae", "mse"])
            #fit
            model.fit(train_data,
                      epochs=num_epochs,
                      verbose=0,
                      validation_data=test_data,
                      callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                                  patience=200,
                                                                  restore_best_weights=True),
                                 tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                                     patience=100,
                                                                     verbose=1)]
                      )
            #append fitted model
            ensemble_models.append(model)
    
    return ensemble_models

#get list of trained ensemble models
ensemble_models=get_ensemble_models(num_iter=5,
                                    num_epochs=1000)

#make prediction 
def make_ensemble_preds(ensemble_models, data):
    ensemble_preds=[]
    for model in ensemble_models:
        preds = model.predict(data)
        ensemble_preds.append(preds)
    return tf.constant(tf.squeeze(ensemble_preds))

ensemble_preds = make_ensemble_preds(ensemble_models=ensemble_models,
                                     data=test_dataset)            
            
            
#evaluate preds
ensemble_mean = tf.reduce_mean(ensemble_preds, axis=0)
ensebmle_median= np.median(ensemble_preds, axis=0)

ensemble_results = evaluate_preds(y_true=y_test,
                                y_pred=ensemble_mean) 
           
    
    
    
#plotting prediction intervals(uncertainty estimates) of our ensemble    
# it would be helpful if we knew a range of where that prediction came from?
#ınsted of 50_000 on the dot usd how about 48000 to 52000
#one way to get %95 confidence prediction intervals for a deep learning is bootstrap method:
    #1. Take predictions from a number of randomly initialized models
    #2. Measure standard deviation of predictipns
    #3. multiply standard deviation by 1.96 ("assumes %95 of observations fall withn 1.96 standard deviations of mean")
    #4 to get upper and lower bound add and subtract the value obtained in 3 to the mean/median of predictions made in 1


#Find upper and lower bounds
def get_upper_lower(preds): #take predictiopns from randomly initialized models
    
    #measure standard deviationss
    std = tf.math.reduce_std(preds, axis=0)
    #get interval
    interval = 1.96*std
    #get prediction interval upper and lower bounds
    preds_mean = tf.reduce_mean(preds, axis=0)
    lower, upper = preds_mean - interval, preds_mean + interval
    
    return lower, upper

#upper and lower bound of %95 pred interval
lower, upper = get_upper_lower(preds=ensemble_preds)
#get the median/mean values of ensemble
ensemble_median=np.median(ensemble_preds, axis=0)
#plot the median of our ensemble preds along with the prediction intervals
offset=500
plt.figure()
plt.plot(X_test.index[offset:], y_test[offset:], "g", label="Test Data")
plt.plot(X_test.index[offset:], ensemble_median[offset:], "-", color="black", label="Ensemble Median")
plt.xlabel("Date")
plt.ylabel("BTC Price")
#to plot upper and lower bounds
plt.fill_between(X_test.index[offset:],
                 (lower)[offset:],
                 (upper)[offset:],label="prediction interval")
plt.legend(loc="upper left", fontsize=5)

#all of predictions is lagging only be aware of that nobody will rich
#Traim a model full historical data and predict future 
#so far we predicted test dataset pseudofuture
HORIZON=1
WINDOW_SIZE=7

bitcoin_prices_windowed.tail()
#train model on entire 
X_all = bitcoin_prices_windowed.dropna().drop(["Price", "block_reward"], axis=1).to_numpy()
y_all = bitcoin_prices_windowed.dropna()["Price"].to_numpy()

len(X_all), len(y_all)

features_dataset_all = tf.data.Dataset.from_tensor_slices(X_all)
labels_dataset_all = tf.data.Dataset.from_tensor_slices(y_all)

#combine features and labels
dataset_all = tf.data.Dataset.zip((features_dataset_all, labels_dataset_all))
BATCH_SIZE=1024
dataset_all = dataset_all.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

tf.random.set_seed(42)
#create model(niceandsimple)
model_9 = tf.keras.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(HORIZON)
    ], name="model_9")

model_9.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.Adam())

model_9.fit(dataset_all,
            epochs=100,
            verbose=0)

#make predictions into future
#how many time steps to predict future?
INTO_FUTURE=14
#to make predictions into the future 
#1. takes input:
    #a list of values 
    #a trained model
    #window into the future to predict (INTOFUTURE)
    #window size : model trained on, the model only predict on trained on
    #creates a emptylist for future forecast and extract the last 'WINDOW_SIZE' from input 
    #Loop 'INTO FUTURE times making predict, remove first value and append new (first because we designed our windows when train.. by dont using make windows)


#1 create function to make predictions into the future
def make_future_forecasts(values, model, into_future, window_size=WINDOW_SIZE) -> list:
    """
    make future forecast into_future steps after value ends.
    returns future forecasts as a list of floats

    """
    #2 create empty list for future forecasts
    future_forecast=[]
    last_window = values[-WINDOW_SIZE:]
    for _ in range(INTO_FUTURE):
        #predict on last window and appent again again aga,n
        future_pred = model_9.predict(tf.expand_dims(last_window, axis=0))
        print(f"Predicting on : \n {last_window} -> Prediction: {tf.squeeze(future_pred).numpy()}\n")
        #append prediction
        future_forecast.append(tf.squeeze(future_pred).numpy())
        #update last window
        last_window=np.append(last_window, future_pred)[-WINDOW_SIZE:]
        
    return future_forecast
    

        
future_forecast = make_future_forecasts(values=y_all,
                                       model=model_9,
                                       into_future=INTO_FUTURE,
                                       window_size=WINDOW_SIZE)     
    
#plot future forecast

def get_future_dates(start_date, into_future, offset=1):
    start_date=start_date +np.timedelta64(offset, "D") #specift start day 
    end_date=start_date + np.timedelta64(into_future, "D") #speficy end date
    return np.arange(start_date, end_date, dtype="datetime64[D]") #return date range

    
    
last_timestep = bitcoin_prices.index[-1]
next_time_steps = get_future_dates(start_date=last_timestep,
                                   into_future=INTO_FUTURE)

next_time_steps

#plot future price predictions
plt.figure()
plot_time_series(bitcoin_prices.index, btc_price, start=2500, format="-", label="actual btc price")
plot_time_series(next_time_steps, future_forecast, format="-", label="predicted btc price")

#insert last timesetp/finalprice into next time steps and future forecast for plot good fine
next_time_steps = np.insert(next_time_steps, 0, last_timestep)
future_forecast = np.insert(future_forecast, 0, btc_price[-1])

plot_time_series(bitcoin_prices.index, btc_price, start=2500, format="-", label="actual btc price")
plot_time_series(next_time_steps, future_forecast, format="-", label="predicted btc price")


#add turkey problem to data
#impactf of highly unlikely
btc_price_turkey = btc_price.copy()
btc_price_turkey[-1] = btc_price_turkey[-1] / 100

btc_timesteps_turkey = np.array(bitcoin_prices.index)

plt.figure()
plot_time_series(timesteps=btc_timesteps_turkey,
                 values=btc_price_turkey,
                 format="-",
                 label="BTC price+turkey",
                 start=2500)

#create train and test set 
full_windows, full_labels = make_windows(np.array(btc_price_turkey), window_size=WINDOW_SIZE, horizon=HORIZON)
X_train, X_test, y_train, y_test = make_train_test_splits(full_windows, full_labels)
len(X_train), len(X_test), len(y_train), len(y_test)

turkey_model = tf.keras.models.clone_model(model_1)
turkey_model._name = "model_10_turkey_model"
turkey_model.compile(loss="mae",
                     optimizer=tf.keras.optimizers.Adam())

turkey_model.fit(X_train, y_train,
                 epochs=100,
                 batch_size=128,
                 validation_data=(X_test, y_test))


