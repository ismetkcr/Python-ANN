# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 00:07:11 2025

@author: ismt
"""
#this is for training purposes only 

import numpy as np

input_data = np.array([
    [1, 2, 3],    # day 1: feature1=1, feature2=2, feature3=3
    [4, 5, 6],    # day 2: feature1=4, feature2=5, feature3=6
    [7, 8, 9],    # day 3: feature1=7, feature2=8, feature3=9
    [10, 11, 12], # day 4: feature1=10, feature2=11, feature3=12
])

print("Input data shape:", input_data.shape)  # (4, 3)
print("Input data:\n", input_data)
print("\n" + "="*50 + "\n")

# Parameters
T = 2  # look back 2 time steps
D = 3  # number of features
N = len(input_data) - T  # number of possible samples (4-2 = 2)

# Create the 3D matrix to store sequences

X = np.zeros((N, T, D))
print("X shape:", X.shape)

for t in range(N):
    X[t, :, :] = input_data[t:t+T]
    print(f"Filling sample {t}...")
    print(f"Taking rows {t} to {t+T} from input_data:")
    print(input_data[t:t+T])
    print(f"\nX[{t}] now contains:")
    print(X[t])
    print("\nCurrent full X:")
    print(X)
    print("\n" + "="*50 + "\n")
    

 #Final result explanation
print("Final X shape:", X.shape)
print("\nX[0] (First sample):")
print(X[0])  # Shows days 1-2
print("\nX[1] (Second sample):")
print(X[1])  # Shows days 2-3