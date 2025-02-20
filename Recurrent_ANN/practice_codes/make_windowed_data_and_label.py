# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:32:25 2025

@author: ismt
"""

#Univariate time series example
import pandas as pd
import numpy as np

#Generate random univariatetime series data
np.random.seed(42)
timesteps=np.arange(100)
prices=np.cumsum(np.random.randn(100)) #random walk for process

#create dataframe
univariate_data = pd.DataFrame({
    "Date": pd.date_range(start="2023-01-01", periods=100, freq="D"),
    "Price": prices
    })


univariate_data["Price"].plot()
univariate_data.set_index("Date", inplace=True)
print(univariate_data.head())


#create windows and labels for univariate data
def make_windows(x, window_size=5, horizon=2):
    window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
    print(f"window_step :\n  {window_step}")
    sliding_indexes = np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T
    print(f"sliding_indexes : \n {sliding_indexes}")

    window_indexes = window_step + sliding_indexes
    print(f"window_indexes : \n {window_indexes}")
    print()
    windowed_array = x[window_indexes]
    windows, labels = windowed_array[:, :-horizon], windowed_array[:, -horizon:]
    return windows, labels


#try function 
x = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190]) 
print(f"univariateData is \n {x}\n")
windows, labels = make_windows(x) 
print(f"windows : \n {windows},\n labels \n {labels}")
WINDOW_SIZE = 5
HORIZON = 2

windows, labels = make_windows(univariate_data["Price"].values, window_size=WINDOW_SIZE, horizon=HORIZON)
print("Windows shape", windows.shape)
print("labels shape:", labels.shape)
print("First window:", windows[0])
print("First label:", labels[0])




#multivariate time series example
import numpy as np
import pandas as pd

np.random.seed(42)
timesteps=np.arange(100) #100timesteps
#prices random walk
prices=np.cumsum(np.random.randn(100))
#volume random values between 100 and 1000
volume = np.random.randint(100, 1000, size=100)
#sentiment vandom values between -1 and 1
sentiment = np.random.uniform(-1, 1, size=100)

#create dataFrame
multivariate_data = pd.DataFrame({
    "Date": pd.date_range(start="2023-01-01", periods=100, freq="D"),
    "Price": prices,
    "Volume": volume,
    "Sentiment": sentiment
    })

# Set Date as index
multivariate_data.set_index("Date", inplace=True)

print(multivariate_data.head())

#creating windows and labels for multivariate data

def make_multivariate_windows(data, window_size=WINDOW_SIZE, horizon=HORIZON):
    windows=[]
    labels=[]
    for i in range(len(data)-(window_size+horizon-1)):
        window = data.iloc[i:i+window_size].values #features for the window
        label = data.iloc[i+window_size : i+window_size+horizon]["Price"].values
        
        windows.append(window)
        labels.append(label)
    return np.array(windows), np.array(labels)

windows, labels = make_multivariate_windows(data=multivariate_data,
                                            window_size=WINDOW_SIZE,
                                            horizon=HORIZON)


print("Windows shape:", windows.shape)
print("Labels shape:", labels.shape)
print("First window (features):", windows[0])
print("First label (target):", labels[0])


#multivariate data adding block feauture

import numpy as np
import pandas as pd

# Sample data
np.random.seed(42)
timesteps = np.arange(100)  # 100 timesteps
prices = np.cumsum(np.random.randn(100))  # prices random walk
volume = np.random.randint(100, 1000, size=100)  # volume random values between 100 and 1000
sentiment = np.random.uniform(-1, 1, size=100)  # sentiment random values between -1 and 1

# Create DataFrame
multivariate_data = pd.DataFrame({
    "Date": pd.date_range(start="2023-01-01", periods=100, freq="D"),
    "Price": prices,
    "Volume": volume,
    "Sentiment": sentiment
})

# Set Date as index
multivariate_data.set_index("Date", inplace=True)

# Add block_reward feature
block_reward_1 = 50  # 3 January 2009
block_reward_2 = 25  # 8 November 2012
block_reward_3 = 12.5  # 9 July 2016
block_reward_4 = 6.25  # 18 May 2020

# Block reward dates
block_reward_2_datetime = np.datetime64("2012-11-28")
block_reward_3_datetime = np.datetime64("2016-07-09")
block_reward_4_datetime = np.datetime64("2020-05-18")

# Create date ranges of where specific block_reward values should be
block_reward_2_days = (block_reward_3_datetime - multivariate_data.index[0]).days
block_reward_3_days = (block_reward_4_datetime - multivariate_data.index[0]).days

# Add block reward values as a feature in DataFrame
multivariate_data["block_reward"] = None
multivariate_data.iloc[:block_reward_2_days, -1] = block_reward_2
multivariate_data.iloc[block_reward_2_days:block_reward_3_days, -1] = block_reward_3
multivariate_data.iloc[block_reward_3_days:, -1] = block_reward_4

# Function to create windows and labels for multivariate data
def make_multivariate_windows(data, window_size=7, horizon=1):
    windows = []
    labels = []
    for i in range(len(data) - (window_size + horizon - 1)):
        window = data.iloc[i:i + window_size].drop(columns=["block_reward"]).values  # Features for the window
        print("first", window.shape)
        block_reward = data.iloc[i + window_size - 1]["block_reward"] 
        # Block reward for the last point in the window
        window = np.append(window, block_reward)  # Append block reward to the window
        print("second", window.shape)
        label = data.iloc[i + window_size : i + window_size + horizon]["Price"].values  # Target
        
        windows.append(window)
        labels.append(label)
    return np.array(windows), np.array(labels)

# Define window size and horizon
WINDOW_SIZE = 7
HORIZON = 1

# Create windows and labels
windows, labels = make_multivariate_windows(data=multivariate_data, window_size=WINDOW_SIZE, horizon=HORIZON)

# Print shapes and examples
print("Windows shape:", windows.shape)
print("Labels shape:", labels.shape)
print("First window (features):", windows[0])
print("First label (target):", labels[0])